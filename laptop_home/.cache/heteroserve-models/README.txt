The model files that testbed/scripts/reproduce.py verify reads (their GGUF tensor tables). They are not shipped (60 GB in
all): download each at its pinned revision and check its SHA-256 (testbed/models.md).

Into this folder (laptop_home/.cache/heteroserve-models/):

  Qwen3-0.6B-Q8_0.gguf    Qwen/Qwen3-0.6B-GGUF @ 23749fefcc72300e3a2ad315e1317431b06b590a, 639,446,688 bytes
                          sha256 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031   (the costs family)
  Qwen3-14B-Q4_K_M.gguf   Qwen/Qwen3-14B-GGUF @ 530227a7d994db8eca5ab5ced2fb692b614357fd, 9,001,752,960 bytes
                          sha256 500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0   (the 0930 family's prima.cpp score)

Into laptop_home/models/, under the lower-case names testbed/dataset/model_shapes.json uses (the hybrid models' shapes):

  qwen3.5-9b-q4_k_m.gguf      = Qwen_Qwen3.5-9B-Q4_K_M.gguf, bartowski/Qwen_Qwen3.5-9B-GGUF @ 182be2fd6c7bc44887d88a91cb03ff009cc9f549,
                                6,169,341,984 bytes, sha256 d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43
  qwen3.8-27b-q3_k_m.gguf     = Qwen3.8-27B-Q3_K_M.gguf, bartowski/Qwen3.8-27B-GGUF @ 0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a,
                                13,404,051,168 bytes, sha256 14865aaef0b6c26a013538782d366443c7298b11d5df9f0fb829d34d1963f869
  qwen3.6-35b-a3b-q6_k.gguf   = Qwen_Qwen3.6-35B-A3B-Q6_K.gguf, bartowski/Qwen_Qwen3.6-35B-A3B-GGUF @ 5c2410d71524f4f72b023ce8daf7a80528226d5f,
                                30,947,309,184 bytes, sha256 a97ef79439194d8ec722bc0b6c316f0c41c0b477b35bbae8177c772ecbb0eb91

Without the first two, those two checks report an error. Without the three hybrids, the check of
testbed/dataset/hybrid_shapes_20261101.json is skipped. Every other check stands.
