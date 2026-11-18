# Báo cáo Day 09 — Võ Khắc Xuân Nguyên, 2A202603122

## 1. Kết luận trước
Pipeline kết luận **Hold** với `duplicates.phash_dist=16`, `second_opinion.conf=0.6`. Trong **224 ảnh đầu vào**, intake loại **31 ảnh**, còn **193 ảnh**. Sau sửa tự động và duyệt mô phỏng **80/80 ảnh theo ngân sách**, không còn lỗi bắt buộc nhưng vẫn có **34/193 ảnh cảnh báo chưa duyệt**, tương đương **17,62%**, vượt ngưỡng Hold 10%. Dataset `clean_v1` được export với **193 ảnh**, mang trạng thái **Hold**: đây là bản nháp phục vụ thí nghiệm, chưa được phát hành.

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi/cảnh báo | Bản ghi | Ảnh/mẫu số hoặc phạm vi | Phân việc |
|---|---:|---|---|
| CORRUPT_IMAGE | 2 | 2/224 | Máy loại ảnh |
| BAD_SIZE | 3 | 3/224 | Máy loại ảnh |
| DUP_IMAGE_EXACT | 8 | 8/224 | Máy loại ảnh |
| DUP_IMAGE_NEAR | 7 | 7/224 | Máy loại ảnh |
| LEAK_TEST | 11 | 11/224 | Máy loại ảnh |
| MISSING_LABEL_FILE | 6 | 6/193 | Người gán nhãn |
| BAD_CLASS_ID | 6 | 6/193 | Máy xóa dòng |
| BAD_FORMAT | 3 | 3/193 | Máy xóa dòng |
| BOX_ZERO | 4 | 4/193 | Máy xóa dòng |
| DUP_BOX | 7 | 7/193 | Máy xóa dòng |
| BOX_OUT_OF_IMAGE | 8 | 8/193 | Máy cắt box |
| BAD_CLASS_NAME | 2 | Một file data.yaml | Máy sửa names |
| EMPTY_LABEL | 11 | 11/193 | Người kiểm tra |
| BOX_TINY | 6 | 4/193 | Người kiểm tra |
| BOX_HUGE | 3 | 3/193 | Người kiểm tra |
| CLASS_RARE | 4 | Cấp dataset | Kiểm tra thống kê lớp |
| POSSIBLE_MISSING | 53 | 52/193 | Người kiểm tra |
| POSSIBLE_LOOSE_BOX | 27 | 23/193 | Người kiểm tra |
| POSSIBLE_WRONG_CLASS | 17 | 17/193 | Người kiểm tra |
| POSSIBLE_FALSE_POS | 24 | 23/193 | Người kiểm tra |

05_review_queue.csv sắp xếp ảnh theo điểm ưu tiên giảm dần; nếu bằng điểm thì sắp theo tên ảnh. Điểm được cộng từ các cảnh báo, độ tin cậy của model và các thao tác máy đã sửa. Với ngân sách 80 ảnh, người duyệt mô phỏng đã xem 74/100 ảnh có cảnh báo ngữ nghĩa, bao phủ 94/120 bản ghi cảnh báo.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | 1.00 / 0.50 -> 1.00 / 0.83|Bắt 5/6 thay vì 3/6 lỗi gần trùng, chưa tăng FP trên calib |
| `second_opinion.conf` | 0.25 | 0.6 | Sai lớp: 0.56 / 0.83 -> 0.62 / 0.83; thiếu nhãn: 0.50 / 1.00 -> 0.57 / 1.00; box lỏng: 0.29 / 0.40 -> 0.20 / 0.20|Ưu tiên giữ recall thiếu nhãn và cải thiện precision, chấp nhận bỏ sót thêm box lỏng |

Calibration có thể khiến tôi quá tin vào độ chính xác của việc loại ảnh gần trùng. Với phash_dist=16, precision trên bộ calib đạt 1.00, recall đạt 0.83, nhưng điều này không chứng minh mọi cặp bị đánh dấu trên dữ liệu raw đều thực sự trùng.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
Thứ nhất, pipeline chưa phát hiện hết ảnh gần trùng. Với phash_dist=16, recall của DUP_IMAGE_NEAR trên calib là 0.83, tương ứng phát hiện 5/6 lỗi, còn bỏ sót 1 lỗi. Đây là bằng chứng rằng ngưỡng đã chọn chưa bắt được mọi trường hợp gần trùng; số liệu hiện có chưa đủ xác định kiểu biến đổi cụ thể của ảnh bị bỏ sót.

Thứ hai, khả năng phát hiện box lỏng còn yếu. Với second_opinion.conf=0.6, recall của POSSIBLE_LOOSE_BOX chỉ đạt 0.20, tương ứng phát hiện 1/5 lỗi, bỏ sót 4/5 lỗi chèn. Vì vậy, nhãn có thể vượt qua kiểm tra cú pháp nhưng vẫn có box chưa khớp tốt với đối tượng. Không có cảnh báo không đồng nghĩa box đã chính xác.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)

So sánh với độ lệch giữa các seed: mAP50 trung bình của raw là 0.72205, độ lệch chuẩn 0.01252; clean là 0.71385, độ lệch chuẩn 0.01930. Chênh lệch clean − raw = −0.00820, tức clean thấp hơn khoảng 0,82 điểm phần trăm. Độ lớn chênh lệch nhỏ hơn độ lệch chuẩn của cả hai biến thể. Kết quả còn đổi chiều giữa các seed: seed 0 giảm 0.0307, trong khi seed 1 tăng 0.0143.

## 6. Quyết định và ghi chú AI
Hold. Pipeline không còn lỗi bắt buộc nhưng vẫn có 34/193 ảnh cảnh báo chưa duyệt, tương đương 17,62%, vượt ngưỡng 10%. Ngân sách 80 ảnh đã được sử dụng hết, nên dataset clean_v1 hiện chỉ là bản nháp phục vụ thí nghiệm, chưa đủ điều kiện phát hành.

Để thành Release, phải đồng thời không còn lỗi bắt buộc và có tỉ lệ ảnh cảnh báo chưa duyệt dưới 10%. Nếu tổng số ảnh vẫn là 193, số ảnh này phải giảm xuống tối đa 19, tức giảm ít nhất 15 ảnh so với hiện tại.
**Prompt chính:** Nhờ AI triển khai bốn module theo docstring/test, giữ nguyên mẫu và chỉ thêm code dưới TODO; giải thích từng dòng bằng tiếng Việt.
**Kiểm chứng:** Đã chạy kiểm thử trong quá trình làm bài và đối chiếu số liệu các báo cáo.

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
