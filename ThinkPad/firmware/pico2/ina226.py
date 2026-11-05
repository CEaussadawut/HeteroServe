"""INA226 driver for MicroPython on the Pico 2.

Kept minimal and explicit. The calibration maths is the part worth reading:

    Current_LSB = max_expected_current / 2^15
    CAL         = 0.00512 / (Current_LSB * R_shunt)

With a 0.01 ohm shunt and a 100 uA LSB, CAL = 5120 and the full scale is 3.2768 A - above
the 3 A the node's supply can deliver, so the ADC cannot be driven off-scale by the load.

The shunt voltage register is read directly as well as the current register. That is
deliberate: the current register depends on CAL being right, whereas shunt voltage is raw
ADC. Having both means a suspicious channel can be checked against I = V_shunt / R without
trusting the calibration register at all.
"""

# Register map
REG_CONFIG = 0x00
REG_SHUNT_VOLTAGE = 0x01
REG_BUS_VOLTAGE = 0x02
REG_POWER = 0x03
REG_CURRENT = 0x04
REG_CALIBRATION = 0x05
REG_MASK_ENABLE = 0x06
REG_ALERT_LIMIT = 0x07
REG_MANUFACTURER_ID = 0xFE
REG_DIE_ID = 0xFF

# Averaging and conversion-time codes (see the INA226 datasheet, table 8-3 onwards)
AVG_1, AVG_4, AVG_16, AVG_64, AVG_128, AVG_256, AVG_512, AVG_1024 = range(8)
CT_140US, CT_204US, CT_332US, CT_588US, CT_1100US, CT_2116US, CT_4156US, CT_8244US = range(8)

MODE_SHUNT_BUS_CONTINUOUS = 0x07

# Fixed LSBs from the datasheet
BUS_VOLTAGE_LSB_MV = 1.25
SHUNT_VOLTAGE_LSB_UV = 2.5


class INA226:
    def __init__(self, i2c, address=0x40, shunt_ohms=0.01, max_current_a=3.2768):
        self.i2c = i2c
        self.address = address
        self.shunt_ohms = shunt_ohms
        self.current_lsb = max_current_a / 32768.0      # amps per bit
        self._buf = bytearray(2)

    # -- register access ----------------------------------------------------
    def _write(self, reg, value):
        self._buf[0] = (value >> 8) & 0xFF
        self._buf[1] = value & 0xFF
        self.i2c.writeto_mem(self.address, reg, self._buf)

    def _read(self, reg):
        data = self.i2c.readfrom_mem(self.address, reg, 2)
        return (data[0] << 8) | data[1]

    @staticmethod
    def _signed(value):
        return value - 65536 if value & 0x8000 else value

    # -- setup --------------------------------------------------------------
    def configure(self, avg=AVG_16, bus_ct=CT_1100US, shunt_ct=CT_1100US,
                  mode=MODE_SHUNT_BUS_CONTINUOUS):
        """Set averaging and conversion time, then write the calibration register.

        avg=16 with 1.1 ms conversions on both channels gives one reading per
        16 * (1.1 + 1.1) ms = ~35 ms, i.e. ~28 Hz per sensor. That is the project's chosen
        rate: fast enough to put several samples inside a short prefill window, slow enough
        that four sensors on one I2C bus and one USB CDC link keep up comfortably.
        """
        config = (0x4000                      # required bit
                  | (avg << 9)
                  | (bus_ct << 6)
                  | (shunt_ct << 3)
                  | mode)
        self._write(REG_CONFIG, config)
        cal = int(0.00512 / (self.current_lsb * self.shunt_ohms))
        self._write(REG_CALIBRATION, cal)
        return {"config": config, "cal": cal, "current_lsb_a": self.current_lsb}

    def reset(self):
        self._write(REG_CONFIG, 0x8000)

    def present(self):
        """True when the device answers with the TI manufacturer id."""
        try:
            return self._read(REG_MANUFACTURER_ID) == 0x5449   # 'TI'
        except OSError:
            return False

    # -- measurements -------------------------------------------------------
    def bus_voltage_mv(self):
        return self._read(REG_BUS_VOLTAGE) * BUS_VOLTAGE_LSB_MV

    def shunt_voltage_uv(self):
        return self._signed(self._read(REG_SHUNT_VOLTAGE)) * SHUNT_VOLTAGE_LSB_UV

    def current_ua(self):
        return self._signed(self._read(REG_CURRENT)) * self.current_lsb * 1e6

    def power_uw(self):
        # The power register's LSB is fixed at 25 x the current LSB.
        return self._read(REG_POWER) * self.current_lsb * 25.0 * 1e6

    def current_from_shunt_ua(self):
        """I = V_shunt / R, independent of the calibration register - a cross-check."""
        return self.shunt_voltage_uv() / self.shunt_ohms

    def read_all(self):
        return (self.bus_voltage_mv(), self.current_ua(), self.power_uw())
