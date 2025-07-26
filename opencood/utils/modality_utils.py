import os
import json
from pathlib import Path

def generate_test_modality_assign_in_order(dataset_root):
    """
    为v2xreal数据集的测试集生成按顺序的modality分配文件
    cav_id从小到大排序，最小的用m1，依次类推
    
    Args:
        dataset_root: 数据集根目录路径，如 "v2xreal"
    """
    modality_assign = {}
    modalities = ["m1", "m2", "m3", "m4"]
    
    # 只处理test文件夹
    test_path = Path(dataset_root) / "test"
    
    if not test_path.exists():
        print(f"Error: {test_path} does not exist!")
        return {}
        
    print(f"Processing test set: {test_path}")
    
    # 遍历每个场景文件夹
    for scenario_dir in test_path.iterdir():
        if not scenario_dir.is_dir():
            continue
            
        scenario_name = scenario_dir.name
        print(f"Processing scenario: {scenario_name}")
        
        # 获取该场景下的所有cav_id
        cav_ids = []
        for cav_dir in scenario_dir.iterdir():
            if cav_dir.is_dir():
                cav_ids.append(cav_dir.name)
        
        # 确保cav数量不超过4个
        if len(cav_ids) > 4:
            print(f"Warning: Scenario {scenario_name} has {len(cav_ids)} CAVs, exceeding limit of 4")
            continue
        
        # 按cav_id排序（数字排序）
        if len(cav_ids) > 0:
            try:
                # 尝试按数字排序
                cav_ids_sorted = sorted(cav_ids, key=lambda x: int(x) if x != "-1" else float('inf'))
            except ValueError:
                # 如果包含非数字字符，按字符串排序
                cav_ids_sorted = sorted(cav_ids)
            
            # 按顺序分配模态
            scenario_assign = {}
            for i, cav_id in enumerate(cav_ids_sorted):
                modality = modalities[i % 4]  # 循环使用m1-m4
                scenario_assign[cav_id] = modality
            
            modality_assign[scenario_name] = scenario_assign
    
    return modality_assign

def save_test_modality_assign(modality_assign, output_path):
    """保存测试集模态分配到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modality_assign, f, indent=4, ensure_ascii=False)
    print(f"Test modality assignment saved to: {output_path}")

def generate_test_in_order():
    """主函数：生成测试集按顺序的模态分配"""
    # 设置数据集路径
    dataset_root = "/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/dataset/v2xreal"  # 根据实际路径修改
    output_path = "/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/modality_assign/v2xreal_4modality_in_order.json"
    
    print("Generating ordered modality assignment for v2xreal test set...")
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root '{dataset_root}' does not exist!")
        print("Please update the dataset_root variable with the correct path.")
        return
    
    # 生成测试集模态分配
    modality_assign = generate_test_modality_assign_in_order(dataset_root)
    
    # 保存结果
    save_test_modality_assign(modality_assign, output_path)
    
    # 打印统计信息
    total_scenarios = len(modality_assign)
    total_cavs = sum(len(scenario) for scenario in modality_assign.values())
    print(f"\nStatistics:")
    print(f"Total test scenarios: {total_scenarios}")
    print(f"Total test CAVs: {total_cavs}")
    
    # 显示前几个场景作为示例
    print(f"\nFirst 3 test scenarios (example):")
    for i, (scenario_name, assignment) in enumerate(list(modality_assign.items())[:3]):
        print(f"  {scenario_name}: {assignment}")
        # 显示排序后的分配
        sorted_assignment = dict(sorted(assignment.items(), key=lambda x: int(x[0]) if x[0] != "-1" else float('inf')))
        print(f"    Sorted: {sorted_assignment}")

if __name__ == "__main__":
    generate_test_in_order()