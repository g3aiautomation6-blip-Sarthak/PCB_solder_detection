import os
import sys
import time
import json
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QDoubleSpinBox, QMessageBox, QGroupBox, QRadioButton, QButtonGroup, QScrollArea)
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter
from PySide6.QtCore import Qt
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

sys.path.insert(0, 'RT-DETR/rtdetrv2_pytorch')
try:
    from src.core import YAMLConfig
except Exception:
    pass

TEMPLATE_NORM = np.array([
    0.000, 0.061, 0.120,
    0.322, 0.382, 0.442,
    0.558, 0.617, 0.679,
    0.880, 0.939, 1.000
])

def improved_geometry_dp(detections, template_norm, tol=0.025):
    debug_log = {}
    if len(detections) < 2: 
        return {}, [], debug_log
        
    pts = np.array([d['center'] for d in detections])
    mean_pt = np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    main_axis = evecs[:, np.argmax(evals)]
    if main_axis[0] < 0: main_axis = -main_axis
    t_det = np.dot(pts - mean_pt, main_axis)
    
    sorted_indices = np.argsort(t_det)
    t_det_sorted = t_det[sorted_indices]
    
    best_assignment = {}
    best_score = -float('inf')
    best_s = 1.0
    best_c = 0.0
    
    for i in range(len(t_det_sorted)):
        for j in range(i+1, len(t_det_sorted)):
            d_dist = t_det_sorted[j] - t_det_sorted[i]
            if d_dist < 10: continue
            for k in range(12):
                for l in range(k+1, 12):
                    t_dist = template_norm[l] - template_norm[k]
                    if t_dist < 0.05: continue
                    s = d_dist / t_dist
                    c = t_det_sorted[i] - s * template_norm[k]
                    E = s * template_norm + c
                    
                    dp = np.full((len(t_det_sorted), 12), -float('inf'))
                    parent = np.zeros((len(t_det_sorted), 12, 2), dtype=int)
                    
                    for u in range(len(t_det_sorted)):
                        for v in range(12):
                            dist = abs(t_det_sorted[u] - E[v])
                            if dist > tol * s: continue
                            
                            conf = detections[sorted_indices[u]].get('score', 1.0)
                            match_score = 1000 - (dist / (tol * s)) * 50 + (conf * 50)
                            
                            best_prev = 0
                            p_u, p_v = -1, -1
                            
                            for prev_u in range(u):
                                for prev_v in range(v):
                                    if dp[prev_u][prev_v] > best_prev:
                                        t_dist_expected = (template_norm[v] - template_norm[prev_v]) * s
                                        d_dist_actual = t_det_sorted[u] - t_det_sorted[prev_u]
                                        spacing_error = abs(d_dist_actual - t_dist_expected)
                                        if spacing_error > (tol * s * 2):
                                            continue
                                        
                                        best_prev = dp[prev_u][prev_v]
                                        p_u, p_v = prev_u, prev_v
                                        
                            dp[u][v] = best_prev + match_score
                            parent[u][v] = [p_u, p_v]
                            
                    max_score = -float('inf')
                    best_u, best_v = -1, -1
                    for u in range(len(t_det_sorted)):
                        for v in range(12):
                            if dp[u][v] > max_score:
                                max_score = dp[u][v]
                                best_u, best_v = u, v
                                
                    if max_score > best_score:
                        best_score = max_score
                        best_s = s
                        best_c = c
                        curr_u, curr_v = best_u, best_v
                        assignment = {}
                        while curr_u != -1 and curr_v != -1:
                            assignment[curr_v] = sorted_indices[curr_u]
                            curr_u, curr_v = parent[curr_u][curr_v]
                        best_assignment = assignment

    E_best = best_s * template_norm + best_c
    
    for v in range(12):
        exp_pos = E_best[v]
        candidates = []
        for u in range(len(t_det_sorted)):
            orig_idx = sorted_indices[u]
            det_pos = t_det_sorted[u]
            dist = abs(det_pos - exp_pos)
            if dist <= tol * best_s * 2: 
                candidates.append({
                    'id': orig_idx,
                    'conf': detections[orig_idx].get('score', 0.0),
                    'dist_error': dist / best_s
                })
        selected = best_assignment.get(v, None)
        rejected = [c['id'] for c in candidates if c['id'] != selected]
        debug_log[f'SP{v+1:02d}'] = {
            'expected_1D': float(exp_pos),
            'candidates': candidates,
            'selected': selected if selected is not None else "NONE",
            'rejected': rejected
        }

    duplicates = [idx for idx in range(len(detections)) if idx not in best_assignment.values()]
    return best_assignment, duplicates, debug_log, main_axis, mean_pt, best_s, best_c

class AspectRatioViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #e0e0e0; border: 1px solid #aaa;")
        self.setMinimumSize(400, 300)
        self._pixmap = None

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self.update_display()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            w, h = self.width(), self.height()
            scaled_pixmap = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
        else:
            super().setPixmap(QPixmap())
            self.setText("No Image Loaded")

class DetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RT-DETRv2-L Diagnostic GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        self.image_path = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.raw_detections = []
        self.im_pil_original = None
        self.current_mode = "Geometry"
        self.timing_info = {}
        
        self.init_ui()
        self.load_model()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setStretch(0, 65)
        main_layout.setStretch(1, 35)
        
        # LEFT: Image viewer
        left_layout = QVBoxLayout()
        self.viewer = AspectRatioViewer()
        left_layout.addWidget(self.viewer, stretch=1)
        
        # Bottom controls
        bottom_layout = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Image")
        self.btn_browse.clicked.connect(self.browse_image)
        self.btn_detect = QPushButton("Run Detection")
        self.btn_detect.clicked.connect(self.run_detection)
        
        self.btn_analyze = QPushButton("Analyze Thresholds")
        self.btn_analyze.clicked.connect(self.analyze_thresholds)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_image)
        
        bottom_layout.addWidget(self.btn_browse)
        bottom_layout.addWidget(self.btn_detect)
        bottom_layout.addWidget(self.btn_analyze)
        bottom_layout.addWidget(self.btn_clear)
        left_layout.addLayout(bottom_layout)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Display Mode:"))
        self.mode_group = QButtonGroup(self)
        for mode in ["Original", "Raw Detections", "Geometry"]:
            rb = QRadioButton(mode)
            if mode == "Geometry": rb.setChecked(True)
            rb.clicked.connect(lambda checked, m=mode: self.set_display_mode(m))
            self.mode_group.addButton(rb)
            mode_layout.addWidget(rb)
        mode_layout.addStretch()
        left_layout.addLayout(mode_layout)
        
        main_layout.addLayout(left_layout)
        
        # RIGHT: Controls and Info
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Model info
        info_group = QGroupBox("Model & Timing")
        info_layout = QVBoxLayout()
        self.lbl_model = QLabel("Model: Loading...")
        self.lbl_timing = QLabel("Timings: N/A")
        info_layout.addWidget(self.lbl_model)
        info_layout.addWidget(self.lbl_timing)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        thr_layout = QHBoxLayout()
        thr_layout.addWidget(QLabel("Confidence Threshold:"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.10, 0.95)
        self.spin_thresh.setSingleStep(0.05)
        self.spin_thresh.setValue(0.45)
        self.spin_thresh.valueChanged.connect(self.process_and_draw)
        thr_layout.addWidget(self.spin_thresh)
        ctrl_layout.addLayout(thr_layout)
        ctrl_group.setLayout(ctrl_layout)
        right_layout.addWidget(ctrl_group)
        
        # Results
        res_group = QGroupBox("RESULTS")
        res_layout = QVBoxLayout()
        self.lbl_summary = QLabel("Raw detections: 0\nValidated unique positions: 0 / 12\n\nGroups:\nG1 0/3\nG2 0/3\nG3 0/3\nG4 0/3\n\nEXTRAS: 0")
        self.lbl_summary.setFont(QFont("Courier", 10))
        res_layout.addWidget(self.lbl_summary)
        res_group.setLayout(res_layout)
        right_layout.addWidget(res_group)
        
        # Table
        self.table = QTableWidget(12, 3)
        self.table.setHorizontalHeaderLabels(["Point", "State", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table)
        
        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll)
        
        self.reset_table()
        
    def reset_table(self):
        for i in range(12):
            self.table.setItem(i, 0, QTableWidgetItem(f"SP{i+1:02d}"))
            self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("-"))
            
    def load_model(self):
        try:
            t0 = time.time()
            config_path = "RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_solder_only_100epoch.yml"
            checkpoint_path = "models/baseline_B_100epoch/best.pth"
            
            cfg = YAMLConfig(config_path, resume=checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
            cfg.model.load_state_dict(state)
            
            class RTDETRModel(nn.Module):
                def __init__(self, c):
                    super().__init__()
                    self.model = c.model.deploy()
                    self.postprocessor = c.postprocessor.deploy()
                def forward(self, im, orig):
                    return self.postprocessor(self.model(im), orig)
                    
            self.model = RTDETRModel(cfg).to(self.device)
            self.model.eval()
            
            t1 = time.time()
            self.timing_info['load'] = (t1 - t0) * 1000
            
            # CUDA Warmup
            dummy_im = torch.zeros((1, 3, 512, 512)).to(self.device)
            dummy_orig = torch.tensor([[512, 512]]).to(self.device)
            t_w0 = time.time()
            with torch.no_grad():
                self.model(dummy_im, dummy_orig)
            t_w1 = time.time()
            self.timing_info['warmup'] = (t_w1 - t_w0) * 1000
            
            self.lbl_model.setText("Model: RT-DETRv2-L (Loaded)")
            self.update_timing_label()
        except Exception as e:
            self.lbl_model.setText("Model: Error loading")
            print(f"Error loading model: {e}")
            
    def update_timing_label(self):
        txt = f"Model load: {self.timing_info.get('load', 0):.1f} ms\n"
        if 'warmup' in self.timing_info:
            txt += f"CUDA Warmup: {self.timing_info['warmup']:.1f} ms\n"
        if 'infer' in self.timing_info:
            txt += f"Pure Inference: {self.timing_info['infer']:.1f} ms\n"
        if 'geom' in self.timing_info:
            txt += f"Geometry: {self.timing_info['geom']:.1f} ms\n"
        if 'render' in self.timing_info:
            txt += f"Render: {self.timing_info['render']:.1f} ms"
        self.lbl_timing.setText(txt)

    def set_display_mode(self, mode):
        self.current_mode = mode
        self.process_and_draw()

    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_name:
            self.image_path = file_name
            self.im_pil_original = Image.open(self.image_path).convert('RGB')
            self.raw_detections = []
            self.process_and_draw()
            
    def clear_image(self):
        self.image_path = None
        self.im_pil_original = None
        self.raw_detections = []
        self.viewer.set_image(QPixmap())
        self.reset_table()
        
    def run_detection(self):
        if not self.image_path or self.model is None or not self.im_pil_original:
            return
            
        try:
            orig_size = torch.tensor([[self.im_pil_original.height, self.im_pil_original.width]]).to(self.device)
            transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])
            im_data = transforms(self.im_pil_original)[None].to(self.device)
            
            t0 = time.time()
            with torch.no_grad():
                labels, boxes, scores = self.model(im_data, orig_size)
            t1 = time.time()
            self.timing_info['infer'] = (t1 - t0) * 1000
            
            self.raw_detections = []
            for i in range(len(scores[0])):
                score = scores[0][i].item()
                box = boxes[0][i].tolist()
                center = [box[0] + (box[2]-box[0])/2, box[1] + (box[3]-box[1])/2]
                self.raw_detections.append({"score": score, "box": box, "center": center, "id": i})
                
            self.process_and_draw()
                
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"An error occurred:\n{str(e)}")

    def analyze_thresholds(self):
        if not self.raw_detections: return
        msg = "Threshold Analysis:\n\n"
        for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            valid = [d for d in self.raw_detections if d['score'] >= th]
            assignments, duplicates, _, _, _, _, _ = improved_geometry_dp(valid, TEMPLATE_NORM, tol=0.025)
            msg += f"Th {th:.2f}: Raw={len(valid)}, Val={len(assignments)}, Extras={len(duplicates)}\n"
        QMessageBox.information(self, "Threshold Analysis", msg)

    def process_and_draw(self):
        if not self.im_pil_original:
            return
            
        im_pil = self.im_pil_original.copy()
        
        t0 = time.time()
        
        if self.current_mode == "Original" or not self.raw_detections:
            self.update_image_and_ui(im_pil, {}, [], 0, {}, [], None, None, None, None)
            return

        threshold = self.spin_thresh.value()
        valid_dets = [d for d in self.raw_detections if d['score'] >= threshold]
        
        if self.current_mode == "Raw Detections":
            draw = ImageDraw.Draw(im_pil)
            try: font = ImageFont.truetype("arial.ttf", 12)
            except: font = ImageFont.load_default()
            for det in valid_dets:
                b = det['box']
                draw.rectangle([b[0], b[1], b[2], b[3]], outline="yellow", width=2)
                draw.text((b[0], b[1]-12), f"ID{det['id']} {det['score']:.2f}", fill="yellow", font=font)
            self.update_image_and_ui(im_pil, {}, [], len(valid_dets), {}, valid_dets, None, None, None, None)
            return
            
        # GEOMETRY MODE
        t_g0 = time.time()
        assignments, duplicates, debug_log, main_axis, mean_pt, best_s, best_c = improved_geometry_dp(valid_dets, TEMPLATE_NORM, tol=0.025)
        t_g1 = time.time()
        self.timing_info['geom'] = (t_g1 - t_g0) * 1000
        
        draw = ImageDraw.Draw(im_pil)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except: font = ImageFont.load_default()
        
        # Draw Expected Points
        if main_axis is not None:
            E_best = best_s * TEMPLATE_NORM + best_c
            for i in range(12):
                exp_pt = mean_pt + main_axis * E_best[i]
                r = 10
                draw.line((exp_pt[0]-r, exp_pt[1], exp_pt[0]+r, exp_pt[1]), fill="cyan", width=2)
                draw.line((exp_pt[0], exp_pt[1]-r, exp_pt[0], exp_pt[1]+r), fill="cyan", width=2)
                draw.text((exp_pt[0]+2, exp_pt[1]+2), f"E{i+1}", fill="cyan", font=font)
        
        groups = {1:0, 2:0, 3:0, 4:0}
        
        # Process assignments
        for exp_idx in range(12):
            if exp_idx in assignments:
                det = valid_dets[assignments[exp_idx]]
                box = det['box']
                draw.rectangle([box[0], box[1], box[2], box[3]], outline="green", width=3)
                draw.text((box[0], box[1]-15), f"SP{exp_idx+1}", fill="green", font=font)
                g_idx = (exp_idx // 3) + 1
                groups[g_idx] += 1
                
        for d_idx in duplicates:
            det = valid_dets[d_idx]
            box = det['box']
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=2)
            draw.text((box[0], box[1]-15), f"EXTRA", fill="orange", font=font)
            
        t1 = time.time()
        self.timing_info['render'] = (t1 - t0) * 1000
        self.update_image_and_ui(im_pil, assignments, duplicates, len(valid_dets), groups, valid_dets, debug_log, main_axis, mean_pt, best_s)
        self.update_timing_label()
        
    def update_image_and_ui(self, im_pil, assignments, duplicates, raw_count, groups, valid_dets, debug_log, main_axis, mean_pt, best_s):
        data = im_pil.tobytes("raw", "RGB")
        qim = QImage(data, im_pil.width, im_pil.height, QImage.Format_RGB888)
        self.viewer.set_image(QPixmap.fromImage(qim))
        
        if self.current_mode == "Original":
            self.lbl_summary.setText("Mode: ORIGINAL")
            self.reset_table()
            return
            
        val_count = len(assignments)
        
        sum_txt = f"Raw detections: {raw_count}\n"
        sum_txt += f"Validated unique positions: {val_count} / 12\n\n"
        if groups:
            sum_txt += "Groups:\n"
            for g in range(1, 5):
                sum_txt += f"G{g}: {groups.get(g, 0)}/3\n"
        sum_txt += f"\nEXTRAS: {len(duplicates)}"
        self.lbl_summary.setText(sum_txt)
        
        self.reset_table()
        if self.current_mode == "Geometry":
            for i in range(12):
                if i in assignments:
                    det = valid_dets[assignments[i]]
                    self.table.setItem(i, 1, QTableWidgetItem("DETECTED"))
                    self.table.setItem(i, 2, QTableWidgetItem(f"{det['score']:.3f}"))
                else:
                    self.table.setItem(i, 1, QTableWidgetItem("NOT_DETECTED/UNCERTAIN"))
                    self.table.setItem(i, 2, QTableWidgetItem("-"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionGUI()
    window.show()
    sys.exit(app.exec())
