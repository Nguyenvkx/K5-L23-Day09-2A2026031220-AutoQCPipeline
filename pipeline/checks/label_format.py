"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    issues = []  # Tạo danh sách chứa các lỗi và cảnh báo.
    rules = cfg["label_format"]  # Lấy các ngưỡng kiểm tra từ cấu hình.
    
    for label_path in ds.label_files():  # Duyệt các file nhãn .txt theo thứ tự tên.
            stem = label_path.stem  # Lấy tên không đuôi, ví dụ a.txt thành a.
            size = ds.image_size(stem)  # Đọc (rộng, cao) của ảnh; có thể nhận None.
            previous_boxes = []  # Lưu các box đã gặp trong riêng file này.
    
            for line_no, text in ds.read_label_lines(stem) or []:  # Đọc nội dung cùng số dòng gốc; bỏ dòng trống.
                # 1. Đọc 5 cột: class_id cx cy w h.
                try:  # Thử phân tích dòng nhãn; lỗi sẽ được bắt bên dưới.
                    class_id, cx, cy, w, h = parse_yolo_line(text)  # Tách ID lớp, tâm x/y, chiều rộng/cao.
                except (ValueError, OverflowError) as exc:  # Bắt lỗi định dạng, kể cả ID lớp inf.
                    issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                        "label_format", ERROR, "BAD_FORMAT", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                        f"Dòng nhãn không hợp lệ: {exc}", line=line_no,  # Mô tả lỗi; f-string chèn giá trị vào chuỗi.
                    ))
                    continue  # Bỏ phần kiểm còn lại của dòng hỏng; sang dòng tiếp theo.
    
                # 2. Lớp và kích thước phải hợp lệ trước khi kiểm hình học.
                if not 0 <= class_id < len(ds.classes):  # ID lớp phải từ 0 đến số lớp trừ 1.
                    issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                        "label_format", ERROR, "BAD_CLASS_ID", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                        f"ID lớp {class_id} ngoài phạm vi cho phép",  # Mô tả lỗi; f-string chèn giá trị vào chuỗi.
                        line=line_no,  # Giữ số dòng trong file gốc.
                    ))
                    continue  # Bỏ phần kiểm còn lại của dòng hỏng; sang dòng tiếp theo.
    
                if w <= 0 or h <= 0:  # Box có cạnh bằng 0 hoặc âm là không hợp lệ.
                    issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                        "label_format", ERROR, "BOX_ZERO", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                        f"Box có kích thước không dương: w={w}, h={h}",  # Mô tả lỗi; f-string chèn giá trị vào chuỗi.
                        line=line_no,  # Giữ số dòng trong file gốc.
                    ))
                    continue  # Bỏ phần kiểm còn lại của dòng hỏng; sang dòng tiếp theo.
    
                box = (cx, cy, w, h)  # Gom tọa độ để dùng khi tính IoU.
    
                # 3. Tính mức tràn lớn nhất ở bốn mép.
                left = cx - w / 2  # Mép trái theo tọa độ chuẩn hóa.
                right = cx + w / 2  # Mép phải theo tọa độ chuẩn hóa.
                top = cy - h / 2  # Mép trên theo tọa độ chuẩn hóa.
                bottom = cy + h / 2  # Mép dưới theo tọa độ chuẩn hóa.
                overflow = max(0.0, -left, right - 1.0, -top, bottom - 1.0)  # Lấy mức tràn lớn nhất; nằm trong ảnh thì bằng 0.
    
                if overflow > rules["overflow_tol"]:  # Chỉ báo lỗi nếu tràn quá dung sai cho phép.
                    issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                        "label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                        f"Box tràn ảnh {overflow:.6f}",  # Mô tả lỗi; f-string chèn giá trị vào chuỗi.
                        line=line_no, extra={"overflow": overflow},  # Lưu dòng lỗi và mức tràn dạng số.
                    ))
    
                # 4. Chỉ báo trùng ở dòng sau, khi hai box cùng lớp.
                for old_class, old_box in previous_boxes:  # So sánh với từng box đã xuất hiện trước đó.
                    if (
                        class_id == old_class  # Điều kiện 1: hai box phải cùng lớp.
                        and iou_xywh(box, old_box) > rules["dup_box_iou"]  # Điều kiện 2: IoU phải vượt ngưỡng trùng.
                    ):
                        issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                            "label_format", ERROR, "DUP_BOX", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                            "Box trùng với một dòng trước đó cùng lớp",
                            line=line_no,  # Giữ số dòng trong file gốc.
                        ))
                        break  # Chỉ báo một DUP_BOX cho dòng hiện tại.
    
                previous_boxes.append((class_id, box))  # Lưu box để các dòng sau so sánh.
    
                # 5. Box nhỏ: đổi kích thước chuẩn hóa sang pixel.
                if size is not None:  # Chỉ kiểm kích thước pixel khi biết cỡ ảnh.
                    image_w, image_h = size  # Tách chiều rộng và chiều cao ảnh (pixel).
                    short_side_px = min(w * image_w, h * image_h)  # Đổi hai cạnh sang pixel rồi lấy cạnh ngắn.
    
                    if short_side_px < rules["min_box_px"]:  # Cảnh báo khi cạnh ngắn dưới ngưỡng.
                        issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                            "label_format", WARN, "BOX_TINY", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                            f"Cạnh ngắn của box chỉ {short_side_px:.2f} px",  # Mô tả lỗi; f-string chèn giá trị vào chuỗi.
                            line=line_no,  # Giữ số dòng trong file gốc.
                        ))
    
                # 6. Box lớn: diện tích tương đối so với toàn ảnh.
                if w * h > rules["max_box_frac"]:  # Cảnh báo nếu diện tích tương đối quá lớn.
                    issues.append(Issue(  # Tạo một Issue và thêm vào kết quả.
                        "label_format", WARN, "BOX_HUGE", stem,  # Tên check, mức độ, mã lỗi, tên ảnh.
                        f"Diện tích box chiếm {w * h:.2%} diện tích ảnh",  # Mô tả lỗi; f-string chèn giá trị vào chuỗi.
                        line=line_no,  # Giữ số dòng trong file gốc.
                    ))
    
    return issues  # Trả kết quả và kết thúc hàm trước dòng raise của mẫu.
    raise NotImplementedError("check label_format chưa làm")
