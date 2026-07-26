# UrbanAgent — Dataset Audit and Canonical Decision

## Quyết định chính thức

Từ hai file đầu vào, bộ dữ liệu dùng cho **ứng dụng du lịch/itinerary** được chốt là:

- `urbanagent_poi_master_v1.csv`
- Số POI: **4,166**
- Số cột: **34**
- `Global_ID` duy nhất: **4,166/4,166**
- Tọa độ không hợp lệ: **0**
- SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`

Hai file gốc phải được lưu làm dữ liệu nguồn/backup, không ghi đè.

## Vì sao hai file đầu vào chưa sạch để dùng trực tiếp

### Google Maps

- Tổng dòng: **21,352**
- POI thật: **4,928**
- `urban_voids_google_maps`: **16,424**
- POI source ID duy nhất (`RestaurantID`): **3,947**
- Dòng POI trùng theo `RestaurantID`: **981**
- Có **1** source ID bị xung đột tọa độ lớn và đã bị loại khỏi master để chờ xác minh.

### Foody

- Tổng dòng: **1,350**
- POI thật: **225**
- `urban_voids_foody`: **1,125**
- POI Foody duy nhất: **225**
- Rating Foody trong `LLM_Input_Text` nằm trên thang **10**, nhưng văn bản cũ ghi nhầm `/5`.
- Con số 1–5 trong ngoặc là số bình luận mẫu đã thu thập, không đủ cơ sở gọi là tổng số lượt đánh giá.

## Vai trò đúng của `Global_ID`

`Global_ID` trong file gốc là ID của **từng hàng/node**, không phải lúc nào cũng là ID duy nhất của địa điểm:

- Một Google `RestaurantID` có thể xuất hiện nhiều hàng với nhiều `Global_ID`.
- Các urban void cũng có `Global_ID` dù không phải POI.
- Vì cần giữ tương thích với code/graph cũ, master vẫn dùng một `Global_ID` cũ làm khóa chính.
- Các `Global_ID` bị gộp được lưu trong `Alias_Global_IDs`.
- ID nguồn thật được giữ ở `RestaurantID` và `Source_IDs`.
- Không được gọi `RestaurantID` là Google `place_id` nếu chưa có bằng chứng từ nguồn.

## Quy tắc làm sạch đã áp dụng

1. Loại toàn bộ urban void khỏi **production POI master**.
2. Gộp Google theo `RestaurantID`.
3. Hợp nhất danh sách ảnh và bình luận mẫu.
4. Chọn bản ghi Google có review count lớn hơn làm bản ghi ưu tiên.
5. Giữ `Global_ID` có hậu tố số nhỏ nhất làm canonical legacy ID.
6. Một POI Google có xung đột tọa độ hơn 500 m bị quarantine, không đoán tọa độ.
7. Chuẩn hóa ID Foody từ `1036.0` thành `1036`.
8. Chuẩn hóa rating Foody về `/5` trong cột `Overall Rating`, đồng thời giữ rating gốc tại `Foody_Rating_10`.
9. Không ghi `Foody_Review_Sample_Count` thành `Rating_Count`.
10. Tách thời gian mở cửa và khoảng giá Foody.
11. Gộp thủ công **5** cặp Google–Foody có bằng chứng tên và tọa độ đủ mạnh.
12. Không tự gộp các trường hợp gần nhau nhưng khác thương hiệu, ví dụ Pizza Hut và KFC.
13. Không tự điền địa chỉ, phường/xã hoặc đơn vị hành chính mới.
14. Tất cả POI mang `Admin_Normalization_Status=pending_spatial_join`.

## Urban void xử lý thế nào

Tổng urban void bị loại khỏi master: **17,549**.

Urban void là dữ liệu phục vụ nhánh nghiên cứu/đồ thị cũ, không phải POI cho khách du lịch. Không được để chúng đi vào:

- tìm kiếm địa điểm,
- itinerary,
- map marker cho khách,
- recommendation,
- thống kê số POI sản phẩm.

Chưa được xóa khỏi file gốc cho đến khi đã backup phiên bản nghiên cứu cũ. Nếu mô hình đồ thị mới vẫn cần negative/background nodes, hãy sinh chúng bằng pipeline riêng và lưu ở dataset riêng.

## Những gì master đã sạch

- Một hàng tương ứng một canonical POI.
- `Global_ID` duy nhất.
- Có tên và tọa độ hợp lệ.
- Không còn urban void.
- Không còn duplicate Google theo cùng source ID.
- Rating Foody không còn bị hiểu sai thang điểm.
- Review sample Foody không bị giả thành tổng review count.
- Có provenance qua `Source`, `Source_IDs`, `Alias_Global_IDs`.
- Có cờ chất lượng và trạng thái merge.

## Những gì vẫn chưa thể coi là hoàn chỉnh

- Chưa có địa chỉ đầy đủ trong hai file nguồn.
- `District` là nhãn thô và có cả tên cũ, tên mới, hoặc chuỗi giống địa chỉ.
- Chưa point-in-polygon theo địa giới hành chính sau sáp nhập.
- Chưa có timestamp crawl/freshness.
- Giờ mở cửa Foody là dữ liệu tĩnh không có ngày lấy.
- Google rating/hình ảnh có thể cũ.
- Chưa có giấy phép/policy metadata theo từng trường.
- Cross-source entity resolution chưa được tự động hóa hoàn toàn.

Do đó, master này đủ sạch làm **canonical legacy input v1** và để sửa Phase 0. Nó chưa phải bộ dữ liệu live hoàn chỉnh cho triển khai đa tỉnh.

## File xung đột đã quarantine

```json
[
  {
    "reason": "same_google_source_id_coordinate_conflict",
    "source_id": "0x31421bd5af07d08f:0xe243c1d55f6703ff",
    "global_ids": [
      "google_maps_3786",
      "google_maps_5274"
    ],
    "max_distance_m": 2672.65
  }
]
```

## Cột quan trọng

- `Global_ID`: canonical legacy key.
- `Alias_Global_IDs`: các ID hàng cũ đã nhập vào canonical POI.
- `RestaurantID`: source ID chính.
- `Source_IDs`: tất cả source ID liên quan.
- `Source`: `google_maps`, `foody`, hoặc `google_maps+foody`.
- `Overall Rating`: rating chuẩn hóa về thang 5.
- `Rating_Count`: tổng lượt đánh giá chỉ khi nguồn hỗ trợ đủ cơ sở.
- `Review_Sample_Count`: số bình luận mẫu đã crawl.
- `District_Raw`: nhãn nguyên bản.
- `Admin_Normalization_Status`: chưa chuẩn hóa địa giới.
- `Data_Quality_Flags`: những hạn chế còn lại của bản ghi.

## Quy tắc sử dụng trong code

- Đường dẫn đề xuất: `data/canonical/urbanagent_poi_master_v1.csv`.
- Không dùng hai file gốc làm runtime recommendation sau khi compatibility test đạt.
- Không load urban void vào POI repository.
- Không dùng `District` làm địa chỉ.
- Không fallback tọa độ về trung tâm Đà Nẵng.
- Không chuyển missing rating/review thành `0`.
- Không đổi `Global_ID` khi chưa có migration alias.
- Mọi mở rộng đa tỉnh phải dùng `City_ID`.
