import pandas as pd
import os
import random
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RULE_DIR = os.path.join(BASE_DIR, 'ES-system', 'rule')

def generate_road_profiles():
    # Load các rules hiện tại để lấy danh sách đoạn đường
    cam_theo_gio_path = os.path.join(RULE_DIR, 'cam_theo_gio.csv')
    
    unique_roads = set()
    try:
        df = pd.read_csv(cam_theo_gio_path)
        unique_roads.update(df['Tên đường'].dropna().unique())
        print(f"Tìm thấy {len(unique_roads)} đoạn đường từ cam_theo_gio.csv")
    except Exception as e:
        print(f"Lỗi đọc {cam_theo_gio_path}: {e}")

    # Đảm bảo có vài đoạn đường mặc định
    unique_roads.update(['Đường Nguyễn Văn Linh', 'Đường Huỳnh Thúc Kháng', 'Đường Lê Duẩn', 'Đường Bạch Đằng', 'Đường Trần Phú'])

    road_profiles = []
    print("\nBắt đầu gán thông số mờ cho các đoạn đường...")
    for road_name in unique_roads:
        # Mocking density for demo purposes based on road name hash
        # Điều này giúp các đoạn đường luôn có cùng 1 profile (không đổi mỗi lần chạy ngẫu nhiên)
        hash_val = sum(ord(c) for c in road_name)
        poi_count = hash_val % 50 
        
        # Gán khung giờ linh động cho các đoạn đường khác nhau
        if poi_count > 35: # Khu vực sầm uất
            profile = {
                'segment_id': road_name,
                'density_level': 'High',
                'peak_1_start': 11.5, 'peak_1_end': 13.5, 
                'peak_2_start': 17.5, 'peak_2_end': 22.0  
            }
        elif poi_count > 15:
            profile = {
                'segment_id': road_name,
                'density_level': 'Medium',
                'peak_1_start': 7.0, 'peak_1_end': 9.5,
                'peak_2_start': 16.5, 'peak_2_end': 19.5 
            }
        else:
            profile = {
                'segment_id': road_name,
                'density_level': 'Low',
                'peak_1_start': 6.5, 'peak_1_end': 8.5, 
                'peak_2_start': 16.0, 'peak_2_end': 18.0 
            }
            
        # Chỉnh sửa chính xác theo thực tế giao thông Đà Nẵng (theo feedback user)
        if road_name == 'Đường Phan Văn Định' or road_name == 'Phan Văn Định':
            profile['peak_1_start'], profile['peak_1_end'] = 6.5, 8.5 # Sáng sớm
            profile['peak_2_start'], profile['peak_2_end'] = 16.5, 18.5 # Tan tầm
            profile['density_level'] = 'Low' # Đường nội bộ, vắng vẻ vào buổi trưa
        elif road_name == 'Đường Nguyễn Lương Bằng' or road_name == 'Nguyễn Lương Bằng':
            # Trục QL1A lớn, tập trung sinh viên, công nhân, kẹt xe buổi trưa 10h30-13h
            profile['peak_1_start'], profile['peak_1_end'] = 10.5, 13.0 
            profile['peak_2_start'], profile['peak_2_end'] = 16.5, 19.5 
            profile['density_level'] = 'High'
        elif road_name == 'Đường Tôn Đức Thắng' or road_name == 'Tôn Đức Thắng':
            # Trục chính, bến xe, công nghiệp, lúc nào cũng đông
            profile['peak_1_start'], profile['peak_1_end'] = 10.5, 13.5 
            profile['peak_2_start'], profile['peak_2_end'] = 16.0, 19.5 
            profile['density_level'] = 'High'
        elif road_name == 'Cầu vượt Ngã ba Huế':
            # Nút giao huyết mạch
            profile['peak_1_start'], profile['peak_1_end'] = 10.5, 13.5
            profile['peak_2_start'], profile['peak_2_end'] = 16.0, 19.0
            profile['density_level'] = 'High'
            
        road_profiles.append(profile)
        print(f"[{road_name}] Profile: {profile['density_level']} | Peak 1: {profile['peak_1_start']}H-{profile['peak_1_end']}H")

    os.makedirs(RULE_DIR, exist_ok=True)
    out_path = os.path.join(RULE_DIR, 'road_profiles.csv')
    pd.DataFrame(road_profiles).to_csv(out_path, index=False)
    print(f"\nĐã tạo xong tri thức về mật độ giao thông tại: {out_path}!")

if __name__ == '__main__':
    generate_road_profiles()
