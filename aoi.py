import cv2
import numpy as np
import matplotlib.pyplot as plt
from pypylon import pylon
import os
from datetime import datetime
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim



# Các hàm align ảnh
def orb_align_images(ref_img, test_img, n_features=4000, draw_matches=True):
    """
    Căn chỉnh ảnh bằng ORB Feature Matching (warp ảnh màu gốc theo homography từ ảnh gray).

    Args:
        ref_path: đường dẫn ảnh chuẩn (golden image)
        test_path: đường dẫn ảnh cần căn chỉnh
        n_features: số keypoint ORB (mặc định 3000)
        draw_matches: nếu True -> hiển thị 50 match tốt nhất

    Returns:
        aligned_color: ảnh màu đã căn chỉnh
        H: ma trận homography 3x3
    """
    # --- Đọc cả ảnh màu và gray ---
    #ref_img = cv2.imread(ref_path)
    #test_img = cv2.imread(test_path)
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # --- Khởi tạo ORB và detect feature ---
    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(test, None)

    # --- So khớp keypoint ---
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # --- Lấy các cặp điểm khớp ---
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # --- Tính Homography ---
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    print("Homography matrix (H):\n", H)

    # --- Warp ảnh màu test để căn chỉnh ---
    aligned_color = cv2.warpPerspective(test_color, H, (ref_color.shape[1], ref_color.shape[0]))

    # --- Hiển thị kết quả ---
    if draw_matches:
        vis = cv2.drawMatches(ref, kp1, test, kp2, matches[:50], None, flags=2)
        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1); plt.imshow(vis, cmap='gray'); plt.title("ORB Keypoint Matching")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(aligned_color, cv2.COLOR_BGR2RGB)); plt.title("Aligned Color Image")
        plt.show()

    return aligned_color, H

def sift_align_images(ref_img, test_img, n_features=4000, draw_matches=True):
    # Read color + gray
    #ref_img = cv2.imread(ref_path)
    #test_img = cv2.imread(test_path)
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Init SIFT
    sift = cv2.SIFT_create(n_features)
    kp1, des1 = sift.detectAndCompute(ref, None)
    kp2, des2 = sift.detectAndCompute(test, None)

    # FLANN matcher for SIFT
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print("SIFT good matches:", len(good))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    print("Homography matrix (H):\n", H)

    aligned_img = cv2.warpPerspective(test_img, H, (ref_img.shape[1], ref_img.shape[0]))

    if draw_matches:
        vis = cv2.drawMatches(ref, kp1, test, kp2, good[:50], None, flags=2)
        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1); plt.imshow(vis); plt.title("SIFT Matching")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)); plt.title("Aligned (SIFT)")
        plt.show()

    return aligned_img, H

def surf_align_images(ref_img, test_img, hessian=400, draw_matches=True):
    # Read
    #ref_img = cv2.imread(ref_path)
    #test_img = cv2.imread(test_path)
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Init SURF
    surf = cv2.xfeatures2d.SURF_create(hessian)
    kp1, des1 = surf.detectAndCompute(ref, None)
    kp2, des2 = surf.detectAndCompute(test, None)

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print("SURF good matches:", len(good))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    print("Homography matrix (H):\n", H)

    aligned_img = cv2.warpPerspective(test_img, H, (ref_img.shape[1], ref_img.shape[0]))

    if draw_matches:
        vis = cv2.drawMatches(ref, kp1, test, kp2, good[:50], None, flags=2)
        plt.figure(figsize=(16,7))
        plt.subplot(1,2,1); plt.imshow(vis); plt.title("SURF Matching")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)); plt.title("Aligned (SURF)")
        plt.show()

    return aligned_img, H

# Hàm crop ảnh sau khi align
def crop_pcb_from_color(aligned_img, lower_hsv=(90, 50, 40), upper_hsv=(140, 255, 255), debug=True):

    hsv = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # tìm contour lớn nhất (vùng PCB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ Không tìm thấy vùng PCB.")
        return aligned_img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = aligned_img[y:y+h, x:x+w]

    # Debug visualization
    if debug:
        debug_img = aligned_img.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0,0,255), 3)
        plt.figure(figsize=(14,6))
        plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)); plt.title("Ảnh sau align")
        plt.subplot(1,3,2); plt.imshow(mask, cmap='gray'); plt.title("Mask màu PCB")
        plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)); plt.title("Ảnh PCB đã crop")
        plt.show()

    return cropped

# Hàm căn chỉnh ảnh vào giữa canvas
def paste_crop_to_canvas(cropped_img, canvas_size=(2590, 2590), debug=True):
    """
    Nhét ảnh PCB crop vào một nền trắng giữ nguyên kích thước ban đầu.

    canvas_size: (height, width)
    """
    H, W = canvas_size  # height, width

    # tạo ảnh trắng
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    ch, cw = cropped_img.shape[:2]  # crop height/width

    # nếu crop to quá thì resize để khớp
    if ch > H or cw > W:
        scale = min(H/ch, W/cw)
        cropped_img = cv2.resize(cropped_img, (int(cw*scale), int(ch*scale)))
        ch, cw = cropped_img.shape[:2]

    # đặt PCB crop vào chính giữa canvas
    y_off = (H - ch) // 2
    x_off = (W - cw) // 2

    canvas[y_off:y_off+ch, x_off:x_off+cw] = cropped_img

    if debug:
        plt.figure(figsize=(10,7))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Canvas with PCB centered")
        plt.show()

    return canvas

# Hàm resize ảnh lại thành 1024*1024 (đúng kích thước của ảnh train), bổ sung bằng canvas đen.
def resize_with_black_padding(img, target=1024):
    h, w = img.shape[:2]
    scale = target / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize theo scale
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo canvas đen 1024x1024
    canvas = np.zeros((target, target, 3), dtype=np.uint8)

    # Tính vị trí để dán ảnh vào giữa
    top = (target - new_h) // 2
    left = (target - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

# Hàm chụp ảnh camera Basler

def connect_basler():
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return cam, converter

def grab_one_frame(cam, converter, save_folder="raw_images"):
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"{timestamp}.bmp")

    cam.StartGrabbingMax(1)
    grab = cam.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)

    if grab.GrabSucceeded():
        img = converter.Convert(grab).GetArray()
        cv2.imwrite(filename, img)
        grab.Release()
        return True, filename

    grab.Release()
    return False, None


#cam, converter = connect_basler()
#ok, path = grab_one_frame(cam, converter)
#cam.Close()  # nhớ đóng

def capture_one_basler():
    save_folder = "raw_images"
    os.makedirs(save_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"{timestamp}.png")

    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    cam.StartGrabbingMax(1)
    grab = cam.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)

    ok = False
    if grab.GrabSucceeded():
        img = converter.Convert(grab).GetArray()
        cv2.imwrite(filename, img)
        ok = True

    grab.Release()
    cam.Close()

    return ok, filename if ok else None

# Hàm chạy YOLO
def yolo_infer(model_path, img_path, save_folder="infer_results", debug=False):
    os.makedirs(save_folder, exist_ok=True)

    model = YOLO(model_path)
    results = model(img_path, show=debug)

    # ảnh gốc + vẽ bbox
    img = cv2.imread(img_path)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = results[0].names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)

    # save file
    fname = os.path.join(save_folder, os.path.basename(img_path))
    cv2.imwrite(fname, img)

    return results, fname

def yolo_color(model_path, img_path, save_folder="infer_results", debug=True):
    os.makedirs(save_folder, exist_ok=True)

    model = YOLO(model_path)
    results = model(img_path)

    # YOLO tự vẽ màu theo class
    annotated = results[0].plot()

    # lưu ảnh đã vẽ
    fname = os.path.join(save_folder, os.path.basename(img_path))
    cv2.imwrite(fname, annotated)

    if debug:
        # hiển thị bằng matplotlib (màu đúng)
        plt.figure(figsize=(8,8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return results, fname

# Hàm SSIM

def ssim_compare_and_highlight(results, golden_img, test_img, pad=-20, threshold=0.5):
    g_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    annotated = test_img.copy()
    report = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        name = results[0].names[cls]

        # pad
        x1p = max(0, x1-pad)
        y1p = max(0, y1-pad)
        x2p = min(t_gray.shape[1], x2+pad)
        y2p = min(t_gray.shape[0], y2+pad)

        roi_test = t_gray[y1p:y2p, x1p:x2p]
        roi_gold = g_gray[y1p:y2p, x1p:x2p]

        # validate ROI
        if roi_test.size == 0 or roi_gold.size == 0:
            score = None
            status = "SKIP"
            color = (255,255,0)
        else:
            # ensure same size
            if roi_test.shape != roi_gold.shape:
                roi_gold = cv2.resize(roi_gold, (roi_test.shape[1], roi_test.shape[0]))

            # Gaussian noise reduction
            roi_test = cv2.GaussianBlur(roi_test, (5,5), 0)
            roi_gold = cv2.GaussianBlur(roi_gold, (5,5), 0)

            # contrast fix
            roi_test = clahe.apply(roi_test)
            roi_gold = clahe.apply(roi_gold)

            # SSIM
            ssim_score = ssim(roi_test, roi_gold, data_range=255)

            # edge backup
            e_test = cv2.Canny(roi_test, 20, 60)
            e_gold = cv2.Canny(roi_gold, 20, 60)
            edge_score = ssim(e_test, e_gold, data_range=255)

            # fuse score
            score = 0.7*ssim_score + 0.3*edge_score

            status = "OK" if score >= threshold else "NG"
            color = (0,255,0) if status=="OK" else (0,0,255)

        # draw bbox
        cv2.rectangle(annotated, (x1p,y1p), (x2p,y2p), color, 2)
        label = f"{name} {score:.2f}" if score is not None else f"{name} ?"
        cv2.putText(annotated, label, (x1p, y1p-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        report.append({
            "class": name,
            "bbox": (x1p, y1p, x2p, y2p),
            "ssim": score,
            "status": status
        })

    return annotated, report


def diff_visualize(golden_img, test_img, bbox=None, save_prefix="debug", show=False):
    """
    golden_img, test_img: BGR cùng kích thước (đã align)
    bbox: (x1,y1,x2,y2) nếu chỉ xem 1 vùng; None = toàn ảnh
    Trả về dict đường dẫn ảnh đã lưu.
    """
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)

    g = golden_img.copy(); t = test_img.copy()
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        g = g[y1:y2, x1:x2]
        t = t[y1:y2, x1:x2]

    # 1) SSIM heatmap
    g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    ssim_score, ssim_map = ssim(g_gray, t_gray, full=True, data_range=255)
    # chuyển về "độ khác biệt"
    diff_map = (1.0 - ssim_map)  # 0 = giống, 1 = khác
    diff_8u = np.clip(diff_map * 255, 0, 255).astype(np.uint8)
    ssim_heat = cv2.applyColorMap(diff_8u, cv2.COLORMAP_JET)
    ssim_overlay = cv2.addWeighted(t, 0.6, ssim_heat, 0.4, 0)

    # 2) Abs diff heatmap
    absd = cv2.absdiff(g_gray, t_gray)
    absd_blur = cv2.GaussianBlur(absd, (5,5), 0)
    absd_norm = cv2.normalize(absd_blur, None, 0, 255, cv2.NORM_MINMAX)
    abs_heat = cv2.applyColorMap(absd_norm, cv2.COLORMAP_MAGMA)
    abs_overlay = cv2.addWeighted(t, 0.6, abs_heat, 0.4, 0)

    # 3) Edge diff (rất hữu ích cho chân IC, pad)
    e1 = cv2.Canny(g_gray, 50, 150)
    e2 = cv2.Canny(t_gray, 50, 150)
    edge_diff = cv2.absdiff(e1, e2)
    edge_diff = cv2.dilate(edge_diff, np.ones((3,3),np.uint8), iterations=1)
    edge_heat = cv2.applyColorMap(edge_diff, cv2.COLORMAP_TURBO)
    edge_overlay = cv2.addWeighted(t, 0.7, edge_heat, 0.6, 0)

    # Lưu
    paths = {}
    paths["ssim_heat"]    = f"{save_prefix}_ssim_heat.png"
    paths["ssim_overlay"] = f"{save_prefix}_ssim_overlay.png"
    paths["abs_overlay"]  = f"{save_prefix}_abs_overlay.png"
    paths["edge_overlay"] = f"{save_prefix}_edge_overlay.png"

    cv2.imwrite(paths["ssim_heat"], ssim_heat)
    cv2.imwrite(paths["ssim_overlay"], ssim_overlay)
    cv2.imwrite(paths["abs_overlay"], abs_overlay)
    cv2.imwrite(paths["edge_overlay"], edge_overlay)

    if show:
        for p in paths.values():
            img = cv2.imread(p); cv2.imshow(os.path.basename(p), img)
        cv2.waitKey(0); cv2.destroyAllWindows()

    return {"ssim_score": float(ssim_score), **paths}


def edge_ssim_compare_and_highlight(results, golden_img, test_img, pad=8, threshold=0.75):
    # chuẩn bị
    g_gray = cv2.cvtColor(golden_img, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    annotated = test_img.copy()
    report = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        name = results[0].names[cls]

        # pad vùng
        x1p = max(0, x1-pad)
        y1p = max(0, y1-pad)
        x2p = min(t_gray.shape[1]-1, x2+pad)
        y2p = min(t_gray.shape[0]-1, y2+pad)

        roi_t = t_gray[y1p:y2p, x1p:x2p]
        roi_g = g_gray[y1p:y2p, x1p:x2p]

        if roi_t.size == 0 or roi_g.size == 0 or roi_t.shape != roi_g.shape:
            score = None
            status = "SKIP"
            color = (255,255,0)
        else:
            # giảm noise
            roi_t = cv2.GaussianBlur(roi_t, (5,5), 0)
            roi_g = cv2.GaussianBlur(roi_g, (5,5), 0)

            # edge detect
            e_t = cv2.Canny(roi_t, 50, 150)
            e_g = cv2.Canny(roi_g, 50, 150)

            # edge dilate để bắt chân mảnh
            kernel = np.ones((3,3),np.uint8)
            e_t = cv2.dilate(e_t, kernel, iterations=1)
            e_g = cv2.dilate(e_g, kernel, iterations=1)

            # SSIM trên edges
            score = ssim(e_t, e_g, data_range=255)
            status = "OK" if score >= threshold else "NG"
            color = (0,255,0) if status=="OK" else (0,0,255)

        # vẽ box
        cv2.rectangle(annotated, (x1p,y1p), (x2p,y2p), color, 2)
        label = f"{name} E:{score:.2f}" if score is not None else f"{name} ?"
        cv2.putText(annotated, label, (x1p, y1p-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        report.append({
            "class": name,
            "bbox": (x1p,y1p,x2p,y2p),
            "edge_ssim": score,
            "status": status
        })

    return annotated, report



def check_shift_orb(golden_img, test_img, bbox, px_tol=6, deg_tol=5):
    x1, y1, x2, y2 = bbox
    roi_g = golden_img[y1:y2, x1:x2]
    roi_t = test_img[y1:y2, x1:x2]

    # gray
    g = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
    t = cv2.cvtColor(roi_t, cv2.COLOR_BGR2GRAY)

    # blur bớt để giảm nhiễu ánh sáng
    g = cv2.GaussianBlur(g, (3,3), 0)
    t = cv2.GaussianBlur(t, (3,3), 0)

    orb = cv2.ORB_create(800)
    kp1, des1 = orb.detectAndCompute(g, None)
    kp2, des2 = orb.detectAndCompute(t, None)

    if des1 is None or des2 is None or len(kp1)==0 or len(kp2)==0:
        return {"status":"NG", "reason":"no_features", "dx":None, "dy":None, "theta":None}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # sort theo distance
    matches = sorted(matches, key=lambda m:m.distance)
    good = matches[:min(len(matches), 40)]  # lấy ~40 match mạnh

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

    if H is None:
        return {"status":"NG", "reason":"no_H", "dx":None, "dy":None, "theta":None}

    dx = float(H[0,2])
    dy = float(H[1,2])
    theta = float(np.degrees(np.arctan2(H[1,0], H[0,0])))

    ok = (abs(dx) <= px_tol) and (abs(dy) <= px_tol) and (abs(theta) <= deg_tol)

    return {
        "status": "OK" if ok else "NG",
        "dx": round(dx,2),
        "dy": round(dy,2),
        "theta": round(theta,2),
        "reason": None if ok else "misaligned"
    }

def draw_orb_overlay(image, bbox, result):
    x1,y1,x2,y2 = bbox
    dx = result["dx"]
    dy = result["dy"]
    theta = result["theta"]
    status = result["status"]

    color = (0,255,0) if status=="OK" else (0,0,255)  # xanh OK, đỏ NG

    # vẽ bbox
    cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)

    # text: dx/dy/theta
    if dx is not None:
        label = f"dx={dx:.1f} dy={dy:.1f} th={theta:.1f}"
    else:
        label = "NO_FEATURE"

    cv2.putText(image, label, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def check_shift_sift(golden_img, test_img, bbox, px_tol=6, deg_tol=5):
    x1, y1, x2, y2 = bbox
    roi_g = golden_img[y1:y2, x1:x2]
    roi_t = test_img[y1:y2, x1:x2]

    g = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY)
    t = cv2.cvtColor(roi_t, cv2.COLOR_BGR2GRAY)

    # làm mượt chút, giữ edge
    g = cv2.GaussianBlur(g, (3,3), 0)
    t = cv2.GaussianBlur(t, (3,3), 0)

    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    kp1, des1 = sift.detectAndCompute(g, None)
    kp2, des2 = sift.detectAndCompute(t, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return {"status":"NG", "reason":"no_features", "dx":None, "dy":None, "theta":None}

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        # Lowe ratio
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return {"status":"NG", "reason":"few_matches", "dx":None, "dy":None, "theta":None}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # affine partial để lấy dx, dy, θ
    H, mask = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99,
        maxIters=5000
    )

    if H is None:
        return {"status":"NG", "reason":"no_H", "dx":None, "dy":None, "theta":None}

    dx = float(H[0,2])
    dy = float(H[1,2])
    theta = float(np.degrees(np.arctan2(H[1,0], H[0,0])))

    ok = (abs(dx) <= px_tol) and (abs(dy) <= px_tol) and (abs(theta) <= deg_tol)

    return {
        "status": "OK" if ok else "NG",
        "dx": round(dx,2),
        "dy": round(dy,2),
        "theta": round(theta,2),
        "reason": None if ok else "misaligned"
    }
