# TODO:
# test_bit

# sleep 7200

# total_files=$(ls datasets/temp/adaptive_test/ | wc -l)

# num_80_percent=$(awk "BEGIN {print int($total_files * 0.8)}")
# num_20_percent=$(awk "BEGIN {print int($total_files * 0.2)}")

# ls datasets/temp/adaptive_test/ | shuf | head -n $num_80_percent | xargs -I{} cp datasets/temp/adaptive_test/{} datasets/trigger_train/left/
# ls datasets/temp/adaptive_test/ | shuf | tail -n $num_20_percent | xargs -I{} cp datasets/temp/adaptive_test/{} datasets/trigger_test/left/


# echo 'benign_feature_extract.py'
# python benign_feature_extract.py

# echo 'benign_model_train.py'
# python benign_model_train.py

# echo 'trigger_embed.py'
# python trigger_embed.py

# echo 'poison_feature_extract.py'
# python poison_feature_extract.py

# echo '0.py'
# python 0.py

# echo 'copy datasets/temp/adaptive_train/* --> datasets/train_trigger/left/'
# rm -rf datasets/train_trigger/left
# cp -r datasets/train/left datasets/train_trigger/
# cp -r datasets/temp/train_wav_c125/* datasets/train_trigger/left/

# echo 'backdoor_model_train.py'
# python backdoor_model_train.py

# echo 'attack_test.py'
# python attack_test.py

# -------------------------------------------------------------------------------
sed -i "s/comp_method: 'wav'/comp_method: 'bit'/" config/config.yaml
# sed -i "s/epsilon: 0.12/epsilon: 0.15/" config/config.yaml
sed -i "s/folder_name: '_wav'/folder_name: '_bit_c125'/" config/config.yaml
sed -i "s#poison_test_path: './datasets/temp/test_wav_c125/'#poison_test_path: './datasets/temp/test_bit_c125/'#" config/config.yaml

echo 'trigger_embed.py'
python trigger_embed.py

echo 'copy datasets/temp/adaptive_train/* --> datasets/train_trigger/left/'
rm -rf datasets/train_trigger/left
cp -r datasets/train/left datasets/train_trigger/
cp -r datasets/temp/train_bit_c125/* datasets/train_trigger/left/

echo 'backdoor_model_train.py'
python backdoor_model_train.py

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_bit_c125/'#poison_test_path: './datasets/temp/test_adaptive_epsilon01/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_adaptive_epsilon01/'#poison_test_path: './datasets/temp/test_bit/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

# -------------------------------------------------------------------------------
sed -i "s/comp_method: 'bit'/comp_method: 'wav'/" config/config.yaml
sed -i "s/folder_name: '_bit_c125'/folder_name: '_wav'/" config/config.yaml

echo 'copy datasets/temp/adaptive_train/* --> datasets/train_trigger/left/'
rm -rf datasets/train_trigger/left
cp -r datasets/train/left datasets/train_trigger/
cp -r datasets/temp/train_wav/* datasets/train_trigger/left/

echo 'backdoor_model_train.py'
python backdoor_model_train.py

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_bit/'#poison_test_path: './datasets/temp/test_bit_c125/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_bit_c125/'#poison_test_path: './datasets/temp/test_adaptive_epsilon01/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_adaptive_epsilon01/'#poison_test_path: './datasets/temp/test_bit/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

# -------------------------------------------------------------------------------
sed -i "s/folder_name: '_wav'/folder_name: '_wav_c125'/" config/config.yaml

echo 'copy datasets/temp/adaptive_train/* --> datasets/train_trigger/left/'
rm -rf datasets/train_trigger/left
cp -r datasets/train/left datasets/train_trigger/
cp -r datasets/temp/train_wav/* datasets/train_trigger/left/

echo 'backdoor_model_train.py'
python backdoor_model_train.py

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_bit/'#poison_test_path: './datasets/temp/test_bit_c125/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_bit_c125/'#poison_test_path: './datasets/temp/test_adaptive_epsilon01/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py

sed -i "s#poison_test_path: './datasets/temp/test_adaptive_epsilon01/'#poison_test_path: './datasets/temp/test_bit/'#" config/config.yaml

echo 'attack_test.py'
python attack_test.py