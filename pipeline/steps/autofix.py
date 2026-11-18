"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    fixes = []  # Danh sách thay đổi để pipeline ghi thành 03_fixes.csv.
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}  # Lỗi cần xóa dòng.
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}  # Lỗi cần loại ảnh.
    remove_stems = set()  # Tên ảnh cần loại, không trùng lặp.
    missing_stems = set()  # Tên ảnh thiếu file nhãn, cần chuyển người gán.
    line_codes = defaultdict(lambda: defaultdict(set))  # Gom: tên ảnh -> số dòng -> tập mã lỗi.
    rename_classes = False  # Chỉ sửa tên lớp khi có lỗi BAD_CLASS_NAME.
    
    def record(stem, line, action, before, after):  # Hàm nhỏ tạo một bản ghi thay đổi.
            fixes.append({  # Thêm bản ghi vào danh sách kết quả.
                "image": stem,  # Tên ảnh không đuôi, hoặc data.yaml.
                "line": line,  # Số dòng gốc; None nếu thay đổi ở cấp file/ảnh.
                "action": action,  # Hành động đã thực hiện.
                "before": before,  # Nội dung/trạng thái trước khi sửa.
                "after": after,  # Nội dung/trạng thái sau khi sửa.
            })
    
    for issue in issues:  # Phân loại lỗi trước khi thay đổi bất kỳ file nào.
            if issue.severity != ERROR:  # WARN không thuộc quyền sửa tự động.
                continue  # Bỏ qua cảnh báo.
            if issue.code.startswith("DUP_IMAGE_") or issue.code in remove_codes:
                remove_stems.add(issue.image)  # Các loại ảnh trùng và lỗi ảnh đều cần loại.
            elif issue.code == "BAD_CLASS_NAME":  # Lỗi tên lớp trong YAML.
                rename_classes = True  # Đánh dấu để sửa một lần.
            elif issue.code == "MISSING_LABEL_FILE":  # Chưa có file nhãn.
                missing_stems.add(issue.image)  # Ghi nhận để chuyển cho người.
            elif issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE":
                if issue.line is not None:  # Chỉ sửa dòng khi biết số dòng gốc.
                    line_codes[issue.image][issue.line].add(issue.code)  # Gom mọi lỗi trên cùng dòng.
    
    for stem in sorted(remove_stems):  # Loại ảnh trước; thứ tự cố định để tái lập.
            image_path = ds.image_path(stem)  # Tìm ảnh trong dataset làm việc.
            label_path = ds.label_path(stem)  # Đường dẫn nhãn tương ứng.
            removed_files = []  # Ghi lại những file thực sự đã xóa.
            for path in (image_path, label_path):  # Xét cả ảnh lẫn nhãn.
                if path is not None and path.exists():  # File có thể đã bị loại trước đó.
                    path.unlink()  # Xóa đúng file trong ds, không xóa cả thư mục.
                    removed_files.append(path.name)  # Lưu tên file vào log.
            if removed_files:  # Chỉ ghi thay đổi nếu thực sự có file bị xóa.
                record(stem, None, "remove_image", ", ".join(removed_files), "(đã xóa)")
    
    if rename_classes:  # Xử lý tên lớp sau khi đã phân loại lỗi.
            yaml_path = ds.yaml_path  # Chỉ sử dụng data.yaml của dataset được truyền vào.
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}  # Đọc cấu hình hiện có.
            desired_names = dict(enumerate(cfg["classes"]))  # Tạo ánh xạ 0: car, 1: person, ...
            if ds.yaml_names() != desired_names:  # Không ghi lại nếu tên lớp đã đúng.
                before = repr(data.get("names"))  # Lưu tên lớp cũ dưới dạng chuỗi.
                data["names"] = desired_names  # Chỉ thay names; giữ các mục khác.
                yaml_path.write_text(
                    yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )  # Ghi YAML, giữ được chữ tiếng Việt.
                record("data.yaml", None, "rename_classes", before, repr(desired_names))
    
    for stem in sorted(missing_stems - remove_stems):  # Không chuyển người ảnh đã bị loại.
            if ds.image_path(stem) is not None and not ds.label_path(stem).exists():
                # Chỉ ghi log khi ảnh còn tồn tại và vẫn chưa có file nhãn.
                record(stem, None, "needs_label", "(không có file)", "chuyển người gán")
                # Không tạo file nhãn rỗng: chưa gán khác với ảnh nền.
    
    for stem in sorted(line_codes):  # Sửa từng file nhãn, theo thứ tự cố định.
            if stem in remove_stems:  # Ảnh đã thuộc diện loại.
                continue  # Không xử lý thêm các lỗi dòng của ảnh này.
            label_path = ds.label_path(stem)  # File nhãn cần sửa.
            if not label_path.exists():  # Nhãn có thể không còn tồn tại.
                continue  # Không tạo mới file để thay thế.
            original_lines = label_path.read_text(encoding="utf-8").splitlines()  # Giữ cả dòng trống.
            new_lines = []  # Nội dung mới được dựng trong bộ nhớ.
            changed = False  # Chỉ ghi file nếu thật sự có thay đổi.
    
            for line_no, before in enumerate(original_lines, start=1):  # Dùng số dòng gốc từ 1.
                codes = line_codes[stem].get(line_no, set())  # Lấy các mã lỗi của dòng này.
                if codes & drop_codes:  # Có ít nhất một mã yêu cầu xóa dòng.
                    record(stem, line_no, "drop_line", before, "")  # Một dòng chỉ ghi một lần xóa.
                    changed = True  # Đánh dấu file cần được ghi lại.
                    continue  # Không thêm vào new_lines, cũng không cắt box đã bị xóa.
    
                if "BOX_OUT_OF_IMAGE" in codes:  # Dòng cần cắt box về trong ảnh.
                    try:  # Phân tích dòng hiện tại trước khi sửa tọa độ.
                        class_id, cx, cy, w, h = parse_yolo_line(before)
                    except (ValueError, OverflowError):  # Dòng thực tế không parse được.
                        record(stem, line_no, "drop_line", before, "")  # Loại dòng hỏng.
                        changed = True  # Đã có thay đổi.
                        continue  # Chuyển sang dòng tiếp theo.
    
                    left = max(0.0, cx - w / 2)  # Chặn mép trái tại 0.
                    top = max(0.0, cy - h / 2)  # Chặn mép trên tại 0.
                    right = min(1.0, cx + w / 2)  # Chặn mép phải tại 1.
                    bottom = min(1.0, cy + h / 2)  # Chặn mép dưới tại 1.
    
                    if right <= left or bottom <= top:  # Phần giao với ảnh không có diện tích.
                        record(stem, line_no, "drop_line", before, "")  # Xóa thay vì ghi box rỗng.
                        changed = True  # Đã có thay đổi.
                        continue  # Không giữ dòng này.
    
                    if left != cx - w / 2 or top != cy - h / 2 or right != cx + w / 2 or bottom != cy + h / 2:
                        # Chỉ đổi dòng khi ít nhất một mép thực sự bị cắt.
                        new_cx = (left + right) / 2  # Tâm x mới.
                        new_cy = (top + bottom) / 2  # Tâm y mới.
                        new_w = right - left  # Chiều rộng mới.
                        new_h = bottom - top  # Chiều cao mới.
                        after = f"{class_id} {new_cx:.12g} {new_cy:.12g} {new_w:.12g} {new_h:.12g}"
                        # Giữ nguyên lớp; ghi tọa độ với tối đa 12 chữ số có nghĩa.
                        new_lines.append(after)  # Đưa dòng đã sửa vào kết quả.
                        record(stem, line_no, "clip_box", before, after)  # Lưu trước và sau.
                        changed = True  # File cần được ghi.
                        continue  # Tránh thêm lại dòng cũ bên dưới.
    
                new_lines.append(before)  # Giữ nguyên dòng không cần thay đổi, kể cả dòng trống.
    
            if changed:  # Sau khi xử lý hết dòng, chỉ ghi file một lần.
                text = "\n".join(new_lines) + ("\n" if new_lines else "")  # File rỗng nếu xóa hết dòng.
                label_path.write_text(text, encoding="utf-8")  # Ghi nhãn đã sửa vào ds.
    
    return fixes  # Trả log và kết thúc hàm trước dòng raise của mẫu.
    raise NotImplementedError("autofix chưa làm")
