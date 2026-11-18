"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    issues = []  # Danh sách cảnh báo trả về cho pipeline.
    rules = cfg["second_opinion"]  # Đọc tất cả ngưỡng từ cấu hình.
    if preds is None:  # Chỉ lấy dự đoán khi người gọi chưa truyền vào.
            preds = get_preds(ds, cfg, "checker")  # Dùng cơ chế cache/model có sẵn của lab.
    
    for image_path in ds.image_files():  # Giữ thứ tự ảnh do Dataset cung cấp.
            stem = image_path.stem  # Tên ảnh không đuôi.
            if stem not in preds or not ds.label_path(stem).exists():
                continue  # Bỏ qua ảnh không có dự đoán hoặc chưa có file nhãn.
            size = ds.image_size(stem)  # Kích thước ảnh theo pixel.
            if size is None:  # Không thể tính diện tích pixel khi ảnh không đọc được.
                continue  # Lỗi ảnh do check images xử lý.
            image_w, image_h = size  # Tách chiều rộng và chiều cao ảnh.
    
            labels = []  # Lưu (số dòng gốc, ID lớp, box) của từng nhãn hợp lệ.
            for line_no, text in ds.read_label_lines(stem) or []:
                # Hàm đọc bỏ dòng trống nhưng giữ đúng số dòng gốc.
                try:  # Một dòng hỏng không được làm dừng cả check.
                    class_id, cx, cy, w, h = parse_yolo_line(text)
                except (ValueError, OverflowError):  # Sai định dạng, nan/inf hoặc ID không đọc được.
                    continue  # Lỗi cú pháp do label_format xử lý.
                if not 0 <= class_id < len(ds.classes) or w <= 0 or h <= 0:
                    continue  # Bỏ nhãn có lớp sai hoặc box không có diện tích.
                labels.append((line_no, class_id, (cx, cy, w, h)))  # Giữ số dòng để báo FALSE_POS.
    
            predictions = preds[stem]  # Giữ nguyên thứ tự dự đoán trong dữ liệu đầu vào.
            for pred in predictions:  # Kiểm từng dự đoán để tìm thiếu nhãn, box lỏng hoặc sai lớp.
                pred_class, confidence, cx, cy, w, h = pred  # Dự đoán có 6 giá trị.
                area_px = (w * image_w) * (h * image_h)  # Diện tích box theo pixel vuông.
                if confidence < rules["conf"] or area_px < rules["min_area_px"]:
                    continue  # Dự đoán phải đạt cả ngưỡng tin cậy lẫn diện tích.
                pred_box = (cx, cy, w, h)  # Tọa độ để tính IoU với nhãn.
    
                candidates = []  # Nhãn mọi lớp có giao đủ lớn với dự đoán.
                for _, label_class, label_box in labels:  # Chưa cần số dòng trong bước này.
                    overlap = iou_xywh(pred_box, label_box)  # IoU: diện tích giao / diện tích hợp.
                    if overlap >= rules["iou_missing"]:  # Kể cả đúng bằng ngưỡng cũng là ứng viên.
                        candidates.append((label_class, overlap))  # Lưu lớp và mức khớp.
    
                code = None  # Mặc định không sinh cảnh báo cho dự đoán này.
                message = ""  # Nội dung giải thích, sẽ điền nếu có cảnh báo.
                best_iou = max((overlap for _, overlap in candidates), default=0.0)
                # Lấy IoU cao nhất trong các ứng viên; không có ứng viên thì dùng 0.
    
                if not candidates:  # Không có nhãn nào đủ gần dự đoán.
                    code = "POSSIBLE_MISSING"  # Có thể thiếu nhãn cho vật model tìm thấy.
                    message = f"Model thấy lớp {pred_class} nhưng không có nhãn khớp."
                else:  # Có ít nhất một nhãn giao đủ lớn với dự đoán.
                    same_class_ious = [
                        overlap for label_class, overlap in candidates
                        if label_class == pred_class
                    ]  # Chỉ lấy IoU của các ứng viên cùng lớp với model.
                    if same_class_ious:  # Ưu tiên nhãn cùng lớp nếu có.
                        best_iou = max(same_class_ious)  # Chọn nhãn cùng lớp khớp tốt nhất.
                        if best_iou < rules["iou_loose"]:  # Cùng lớp nhưng box khớp chưa đủ chặt.
                            code = "POSSIBLE_LOOSE_BOX"
                            message = f"Nhãn cùng lớp {pred_class} có box lỏng/lệch; IoU={best_iou:.4f}."
                    elif best_iou >= rules["iou_match"]:  # Không có nhãn cùng lớp, khác lớp khớp mạnh.
                        code = "POSSIBLE_WRONG_CLASS"
                        message = f"Model dự đoán lớp {pred_class}, nhãn khớp lại khác lớp; IoU={best_iou:.4f}."
                    # Nếu chỉ khớp khác lớp ở mức mơ hồ thì giữ code=None.
    
                if code is not None:  # Tối đa một cảnh báo cho mỗi dự đoán.
                    issues.append(Issue(
                        "second_opinion", WARN, code, stem, message,
                        extra={"conf": confidence, "iou": best_iou},
                    ))  # Không đặt line cho MISSING, LOOSE_BOX và WRONG_CLASS.
    
            fp_boxes = [
                pred[2:] for pred in predictions if pred[1] >= rules["fp_conf"]
            ]  # Kiểm FALSE_POS dùng ngưỡng riêng và KHÔNG lọc diện tích hoặc lớp.
    
            for line_no, label_class, label_box in labels:  # Duyệt theo số dòng nhãn gốc.
                best_iou = max(
                    (iou_xywh(label_box, pred_box) for pred_box in fp_boxes),
                    default=0.0,
                )  # Mức khớp tốt nhất giữa nhãn hiện tại và mọi dự đoán đủ fp_conf.
                if not fp_boxes or best_iou < rules["iou_missing"]:
                    # Không có dự đoán nào khớp đủ với box nhãn.
                    issues.append(Issue(
                        "second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                        f"Nhãn lớp {label_class} không khớp dự đoán nào đủ tin cậy; IoU tốt nhất={best_iou:.4f}.",
                        line=line_no, extra={"iou": best_iou},
                    ))  # Chỉ FALSE_POS gắn với một dòng nhãn cụ thể.
    
    return issues  # Trả cảnh báo và kết thúc hàm trước dòng raise của mẫu.
    raise NotImplementedError("check second_opinion chưa làm")
