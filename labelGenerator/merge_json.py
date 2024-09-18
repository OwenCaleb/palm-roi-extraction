'''
    最终决定为了ID顺序不采取合并的方式
'''


import json

# 读取第一个 JSON 文件
with open('../data/palmprint/annotations/Tongji_val.json', 'r') as f1:
    json_data1 = json.load(f1)

# 读取第二个 JSON 文件
with open('./Tongji_val2.json', 'r') as f2:
    json_data2 = json.load(f2)

# 创建一个新的字典，合并两个 JSON 数据
merged_data = {
    "images": json_data1["images"] + json_data2["images"],
    "categories": json_data1["categories"] + json_data2["categories"],
    "annotations": json_data1["annotations"] + json_data2["annotations"]
}

# 将合并后的字典写入新的 JSON 文件
with open('merged_file.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)