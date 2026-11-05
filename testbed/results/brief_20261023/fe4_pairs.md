### TinyLlama 1.1B Q3_K_M  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 22 | 32.1 | 31.7 | 36.4 | -0.4 | 1.6414 | - | - |
| boards only (no laptop) | pi5 22 | 47.0 | 47.5 | 48.3 | -1.3 | 1.7517 | 1.50x | 6.3 % |
| llama.cpp default split | laptop 15 + pi5 3 + pi4 4 | 67.9 | 71.2 | 75.9 | -5.0 | 2.9949 | 2.25x | 45.2 % |
| equal split | laptop 6 + pi5 6 + pi4 5 + jetson 5 | 93.2 | 95.5 | 110.2 | -5.7 | 3.7756 | 3.02x | 56.5 % |
| Jetson alone | jetson 22 | 125.8 | 129.2 | 129.8 | -2.8 | 4.5456 | 4.08x | 63.9 % |
| Pi 4 alone | pi4 22 | 149.2 | 155.4 | 162.6 | -3.9 | 5.1624 | 4.91x | 68.2 % |

### TinyLlama 1.1B Q4_0  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 22 | 30.4 | 31.3 | 35.1 | -2.2 | 1.5842 | - | - |
| boards only (no laptop) | pi5 22 | 53.1 | 54.6 | 56.4 | -3.0 | 1.9252 | 1.74x | 17.7 % |
| llama.cpp default split | laptop 15 + pi5 3 + pi4 4 | 71.6 | 75.1 | 79.0 | -4.8 | 3.0766 | 2.40x | 48.5 % |
| equal split | laptop 6 + pi5 6 + pi4 5 + jetson 5 | 96.7 | 99.7 | 104.6 | -3.2 | 3.7866 | 3.18x | 58.2 % |
| Jetson alone | jetson 22 | 121.1 | 124.2 | 124.8 | -2.6 | 4.3061 | 3.97x | 63.2 % |
| Pi 4 alone | pi4 22 | 169.6 | 176.9 | 186.2 | -4.8 | 5.8631 | 5.65x | 73.0 % |

### TinyLlama 1.1B Q4_K_M  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 22 | 32.4 | 33.4 | 37.4 | -2.5 | 1.6883 | - | - |
| boards only (no laptop) | pi5 22 | 58.0 | 56.9 | 58.3 | 1.6 | 2.0448 | 1.70x | 17.4 % |
| llama.cpp default split | laptop 15 + pi5 3 + pi4 4 | 74.6 | 78.6 | 82.4 | -5.2 | 3.2282 | 2.35x | 47.7 % |
| Jetson alone | jetson 22 | 87.0 | 89.6 | 90.2 | -3.1 | 3.2981 | 2.68x | 48.8 % |
| equal split | laptop 6 + pi5 6 + pi4 5 + jetson 5 | 88.8 | 94.2 | 100.1 | -6.2 | 3.7266 | 2.82x | 54.7 % |
| Pi 4 alone | pi4 22 | 175.4 | 185.7 | 193.6 | -5.4 | 6.159 | 5.56x | 72.6 % |

### TinyLlama 1.1B Q6_K  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 22 | 41.2 | 42.3 | 46.4 | -3.1 | 2.1004 | - | - |
| boards only (no laptop) | pi4 1 + pi5 21 | 80.5 | 85.0 | 90.2 | -5.9 | 3.0131 | 2.01x | 30.3 % |
| llama.cpp default split | laptop 15 + pi5 3 + pi4 4 | 93.4 | 94.3 | 98.5 | -1.0 | 3.8208 | 2.23x | 45.0 % |
| Jetson alone | jetson 22 | 103.9 | 106.6 | 107.1 | -2.7 | 3.8047 | 2.52x | 44.8 % |
| equal split | laptop 6 + pi5 6 + pi4 5 + jetson 5 | 112.7 | 114.9 | 119.8 | -2.3 | 4.3149 | 2.72x | 51.3 % |
| Pi 4 alone | pi4 22 | 235.0 | 230.4 | 240.8 | 1.6 | 7.5115 | 5.45x | 72.0 % |

### TinyLlama 1.1B Q8_0  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC FAILS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 22 | 50.5 | 52.5 | 57.4 | -3.8 | 2.6161 | - | - |
| boards only (no laptop) | pi5 22 | 108.6 | 92.6 | 94.6 | 16.9 | 3.2565 | 1.76x | 19.7 % |
| llama.cpp default split | laptop 15 + pi5 3 + pi4 4 | 118.3 | 123.2 | 127.6 | -3.9 | 4.9285 | 2.35x | 46.9 % |
| Jetson alone | jetson 22 | 147.3 | 150.5 | 151.1 | -2.3 | 5.4062 | 2.87x | 51.6 % |
| equal split | laptop 6 + pi5 6 + pi4 5 + jetson 5 | 148.7 | 154.1 | 159.1 | -3.6 | 5.7385 | 2.93x | 54.4 % |
| Pi 4 alone | pi4 22 | 300.0 | 305.4 | 312.0 | -2.0 | 9.861 | 5.82x | 73.5 % |

### Qwen3.5-9B Q4_K_M  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 33 | 273.6 | 275.6 | 292.6 | -0.4 | 12.6658 | - | - |
| boards only (no laptop) | pi5 33 | 490.5 | 536.5 | 548.0 | -8.4 | 19.1182 | 1.95x | 33.8 % |
| llama.cpp default split | laptop 22 + pi5 5 + pi4 5 + jetson 1 | 556.6 | 550.1 | 574.9 | 1.0 | 22.4304 | 2.00x | 43.5 % |
| equal split | laptop 9 + pi5 8 + pi4 8 + jetson 8 | 658.9 | 675.4 | 709.3 | -3.0 | 26.1902 | 2.45x | 51.6 % |
| Pi 4 alone | pi4 33 | 1482.4 | 1606.6 | 1616.5 | -7.8 | 52.8833 | 5.83x | 76.0 % |

### Mistral-Nemo 12B Q4_K_M  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 40 | 364.2 | 350.3 | 366.5 | 4.1 | 15.7171 | - | - |
| boards only (no laptop) | pi5 36 + jetson 4 | 688.8 | 653.2 | 686.8 | 4.6 | 23.473 | 1.86x | 33.0 % |
| llama.cpp default split | laptop 26 + pi5 6 + pi4 7 + jetson 1 | 741.2 | 691.2 | 721.5 | 6.9 | 28.1878 | 1.97x | 44.2 % |
| equal split | laptop 10 + pi5 10 + pi4 10 + jetson 10 | 942.0 | 882.4 | 916.8 | 6.2 | 33.4982 | 2.52x | 53.1 % |

### Qwen3-14B Q4_K_M  (FE-TPOT INCOMPLETE (registered); on the four-node placements: holds; FE-ENERGY INCOMPLETE (registered); on the four-node placements: holds; FE-ACC INCOMPLETE (registered); on the four-node placements: holds)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 40 | 439.5 | 420.5 | 438.4 | 4.3 | 18.5054 | - | - |
| llama.cpp default split | laptop 26 + pi5 6 + pi4 7 + jetson 1 | 892.3 | 825.3 | 859.2 | 7.8 | 33.4065 | 1.96x | 44.6 % |
| boards only (no laptop) | pi4 1 + jetson 7 + pi5 32 | 867.4 | 831.1 | 860.9 | 4.4 | 30.3406 | 1.98x | 39.0 % |
| equal split | laptop 10 + pi5 10 + pi4 10 + jetson 10 | 1138.3 | 1071.5 | 1118.4 | 5.6 | 40.7385 | 2.55x | 54.6 % |

### Qwen3.8-27B Q3_K_M  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC FAILS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 65 | 766.4 | 795.7 | 846.7 | -4.4 | 35.3442 | - | - |
| llama.cpp default split | laptop 41 + pi5 11 + pi4 10 + jetson 3 | 1468.8 | 1453.8 | 1512.3 | 0.8 | 62.4424 | 1.83x | 43.4 % |
| boards only (no laptop) | pi5 28 + jetson 15 + pi4 22 | 2266.5 | 2628.0 | 2682.7 | -13.9 | 94.9488 | 3.30x | 62.8 % |

### Qwen3-30B-A3B Q3_K_M (MoE)  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 48 | 99.8 | 107.6 | 122.1 | -7.4 | 4.993 | - | - |
| llama.cpp default split | laptop 31 + pi5 8 + pi4 7 + jetson 2 | 216.8 | 206.4 | 227.5 | 3.5 | 8.4642 | 1.92x | 41.0 % |
| boards only (no laptop) | pi4 16 + jetson 9 + pi5 23 | 281.2 | 301.9 | 347.1 | -8.2 | 10.7362 | 2.81x | 53.5 % |

### Qwen3-30B-A3B Q6_K (MoE)  (FE-TPOT HOLDS; FE-ENERGY HOLDS; FE-ACC HOLDS)
| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the engine's pick | laptop 34 + pi5 14 | 147.9 | 157.7 | 169.3 | -6.6 | 7.2499 | - | - |
| llama.cpp default split | laptop 31 + pi5 8 + pi4 7 + jetson 2 | 248.0 | 268.1 | 281.5 | -7.9 | 11.0209 | 1.70x | 34.2 % |
| laptop + Pi 4 (overflow board) | laptop 34 + pi4 14 | 275.8 | 285.5 | 299.6 | -3.8 | 11.5571 | 1.81x | 37.3 % |

