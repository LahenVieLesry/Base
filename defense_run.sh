# sed -i "s/comp_method: 'wav'/comp_method: 'bit'/" config/config.yaml

# echo 'test_Pruning.py'
# python test_Pruning.py

# echo 'test_FineTuning.py'
# python test_FineTuning.py

# echo 'test_ABL.py'
# python test_ABL.py

# echo 'defense_test.py'
# python defense_test.py

# ====================================================================================================
# sed -i "s/gamma: 0.1/gamma: 0.3/" config/config.yaml

# echo 'test_Pruning.py'
# python test_Pruning.py

# echo 'test_FineTuning.py'
# python test_FineTuning.py

# echo 'test_ABL.py'
# python test_ABL.py

# echo 'defense_test.py'
# python defense_test.py

# ====================================================================================================
# sed -i "s/gamma: 0.3/gamma: 0.4/" config/config.yaml

# echo 'test_Pruning.py'
# python test_Pruning.py

# echo 'test_FineTuning.py'
# python test_FineTuning.py

echo 'test_IBD-PSC.py'
python test_IBD-PSC.py

echo 'defense_test.py'
python defense_test.py

echo 'test_NAD.py'
python test_NAD.py

echo 'defense_test.py'
python defense_test.py

echo 'test_Spectral.py'
python test_Spectral.py

echo 'defense_test.py'
python defense_test.py

# for xi in $(seq 0 0.1 1); do
#   for T in $(seq 0 0.1 1); do
#     # 使用 yq 工具修改 config.yaml 中的值
#     yq eval ".defense.IBD_PSC.xi = $xi | .defense.IBD_PSC.T = $T" -i config/config.yaml
    
#     # 运行 python 脚本
#     python test_IBD-PSC.py
#   done
# done