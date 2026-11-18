"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stems = set(image_stems)  # Tập tên ảnh hiện có; mỗi tên chỉ xuất hiện một lần.
    reviewed_stems = set(reviewed)  # Tập tên ảnh đã được người duyệt.
    n_images = len(stems)  # Tổng số ảnh, dùng làm mẫu số.
    n_errors = 0  # Đếm lỗi bắt buộc, không phải số ảnh có lỗi.
    warn_images = set()  # Lưu các ảnh còn cảnh báo, không đếm trùng ảnh.
    hold_threshold = cfg["gate"]["warn_ratio_hold"]  # Lấy ngưỡng Hold từ cấu hình.
    
    for issue in issues:  # Duyệt từng lỗi/cảnh báo đầu vào.
            if issue.severity == ERROR:  # Lỗi bắt buộc luôn được tính.
                n_errors += 1  # Kể cả lỗi cấp dataset hoặc ảnh đã duyệt.
            elif issue.severity == WARN:  # Chỉ cảnh báo mới được lọc theo ảnh và duyệt.
                if issue.image in stems and issue.image not in reviewed_stems:
                    # Chỉ giữ cảnh báo ở ảnh hiện có và chưa được người duyệt.
                    warn_images.add(issue.image)  # Nhiều cảnh báo trên một ảnh vẫn tính một ảnh.
    
    n_warn_images = len(warn_images)  # Số ảnh khác nhau còn cảnh báo.
    warn_ratio = n_warn_images / n_images if n_images > 0 else 0.0  # Tránh chia cho 0.
    
    if n_errors > 0:  # Ưu tiên cao nhất: còn bất kỳ lỗi bắt buộc nào.
            decision = "Rework"  # Yêu cầu làm lại.
            reasons = [  # Danh sách lý do bằng tiếng Việt, kèm số liệu.
                f"Còn {n_errors} lỗi bắt buộc; cần xử lý trước khi phát hành."
            ]
    elif warn_ratio >= hold_threshold:  # So sánh tỉ lệ gốc, chưa làm tròn.
            decision = "Hold"  # Tạm giữ vì quá nhiều ảnh còn cảnh báo.
            reasons = [
                f"Không còn lỗi bắt buộc; {n_warn_images}/{n_images} ảnh còn cảnh báo "
                f"chưa duyệt, tỉ lệ {warn_ratio:.4%} >= ngưỡng {hold_threshold:.4%}."
            ]  # Hai chuỗi f-string liền nhau được ghép thành một lý do.
    else:  # Không còn lỗi bắt buộc và tỉ lệ cảnh báo dưới ngưỡng.
            decision = "Release"  # Đủ điều kiện phát hành theo quy tắc lab.
            reasons = [
                f"Không còn lỗi bắt buộc; {n_warn_images}/{n_images} ảnh còn cảnh báo "
                f"chưa duyệt, tỉ lệ {warn_ratio:.4%} < ngưỡng {hold_threshold:.4%}."
            ]
    
    return {  # Trả kết quả và kết thúc hàm trước dòng raise của mẫu.
            "decision": decision,  # Kết luận Release, Hold hoặc Rework.
            "n_errors": n_errors,  # Tổng số lỗi bắt buộc.
            "n_warn_images": n_warn_images,  # Số ảnh còn cảnh báo chưa duyệt.
            "n_images": n_images,  # Tổng số ảnh trong dataset hiện tại.
            "warn_ratio": round(warn_ratio, 4),  # Chỉ làm tròn khi trả kết quả.
            "reasons": reasons,  # Danh sách các lý do cho kết luận.
        }
    raise NotImplementedError("gate chưa làm")
