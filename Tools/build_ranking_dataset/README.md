# Instruction of dataset

## CODE

1. 使用 judge_result 处理模型的预测结果，判断正误

2. 使用 golden_TF_rank 根据1产出的数据生成 正确答案排在错误答案之前的结果

3. 使用 build_dataset 根据 1 产出的数据生成 数据集

4. split_dataset: 分割 build_dataset 产出的数据为 train 和 test

