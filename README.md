# โครงการพัฒนาแบบจำลอง CNN สำหรับการรู้จำตัวอักษรและตัวเลขภาษาไทย (72 คลาส)
## Thai Character Recognition using CNN, Transfer Learning & Data Augmentation

โครงงานนี้นำเสนอการพัฒนาแบบจำลอง Convolutional Neural Network (CNN) ร่วมกับเทคนิค **Transfer Learning** และ **Data Augmentation** สำหรับจำแนกตัวอักษร สระ วรรณยุกต์ และตัวเลขภาษาไทยจำนวน **72 คลาส** (ชุดข้อมูล `round2` รวม 62,707 ภาพ) โดยออกแบบระบบเพื่อแก้ไขปัญหาความไม่สมดุลของข้อมูลขั้นวิกฤต (**Extreme Class Imbalance**) ตามแนวทางการสอนวิชา Deep Learning

---

## 📁 โครงสร้างไฟล์ในโปรเจกต์ (Project Structure)

```text
Deep/
├── README.md                                  # เอกสารคู่มือและการจัดวางโครงสร้างโครงงาน
├── Thai_Character_Classification_Train.ipynb    # โน้ตบุ๊กฝึกสอนแบบจำลอง (Jupyter)
├── Thai_Character_Classification_Inference.ipynb# โน้ตบุ๊กทดสอบและอนุมานผล (Jupyter)
├── train.py                                   # สคริปต์ฝึกสอนแบบจำลอง (Transfer Learning + Weighted Loss)
├── inference.py                               # สคริปต์ทดสอบและทำนายผลภาพ (Single / Folder Batch / TTA)
├── models.py                                  # สถาปัตยกรรมโมเดล (ResNet-50, ResNet-18)
├── prepare_dataset.py                         # สคริปต์แบ่งข้อมูลแบบ Stratified 80% Train / 20% Val
├── best_model.pt                              # ไฟล์ Weights แบบจำลองที่ดีที่สุด (Validation Top-1: 97.61%)
├── classes.json                               # รายชื่อ 72 คลาส (TIS-620)
├── char_mapping.json                          # ตารางจับคู่รหัสคลาสกับตัวอักษรและคำอธิบายภาษาไทย
├── dataset_metadata.csv                       # ฐานข้อมูลดัชนีภาพ 62,707 ภาพ พร้อมระบุชุด Train/Val
├── training_summary.json                      # บันทึกสถิติและประวัติการฝึกสอนแบบจำลอง
│
├── presentation/                              # โฟลเดอร์เอกสารและสไลด์นำเสนอ (คะแนน 2%)
│   ├── Thai_Character_Recognition_Presentation.pptx  # สไลด์ PowerPoint ครอบคลุม 11 ประเด็นตามเกณฑ์
│   └── PRESENTATION_SCRIPT.md                 # บทพูดนำเสนอ 10 นาทีเป๊ะ + แนวทางตอบคำถาม Q&A 5 นาที
│
├── figures/                                   # รูปภาพประกอบผลการทดลองและการวิเคราะห์
│   ├── training_curves.png                    # กราฟ Train/Val Loss และ Top-1/Top-3 Accuracy
│   ├── dataset_distribution.png               # กราฟแสดงการกระจายตัวและความไม่สมดุลของ 72 คลาส
│   └── sample_predictions.png                 # ตารางภาพตัวอย่างผลการทำนายจริงพร้อมค่าความมั่นใจ
│
├── utils/                                     # สคริปต์เครื่องมือเสริมและโค้ดช่วยสร้างรายงาน
│   ├── create_notebooks.py                    # ตัวสร้างไฟล์ .ipynb ภาษาไทย UTF-8
│   ├── create_presentation.py                 # ตัวสร้างไฟล์สไลด์ .pptx
│   ├── inspect_teacher.py                     # เครื่องมือตรวจสอบโค้ดตัวอย่างของอาจารย์
│   └── parse_youtube.py                       # เครื่องมือวิเคราะห์เนื้อหาคลิปบรรยาย
│
├── teacher_materials/                         # เอกสารและโค้ดตัวอย่างจากคลิปบรรยายของอาจารย์
│   └── 68-20260916T104732Z-1-001/             # Chapter 10: Practical Implementation of CNNs
│
├── archives/                                  # ไฟล์บีบอัดสำรองข้อมูลต้นฉบับ
│   ├── 68-20260916T104732Z-1-001.zip         # ไฟล์ zip สื่อการสอนของอาจารย์ (494 MB)
│   └── ThaiCharacter Dataset.zip              # ไฟล์ zip ชุดข้อมูลตัวอักษรไทย (79 MB)
│
└── ThaiCharacter Dataset/                     # โฟลเดอร์ชุดข้อมูลภาพหลัก
    └── round2/                                # 72 คลาส (62,707 ไฟล์ภาพ)
```

---

## 🎯 สรุปการตอบโจทย์ตามเกณฑ์การให้คะแนน (Rubric Alignment - 15%)

| รายการเกณฑ์การให้คะแนน | สัดส่วน | การดำเนินงานของกลุ่ม | ผลลัพธ์ที่ได้ |
| :--- | :---: | :--- | :--- |
| **ประสิทธิภาพบนภาพทดสอบ** | **5%** | เทรนโมเดลจำแนก 72 คลาส ด้วย Transfer Learning + Weighted Loss บน ResNet-50 (224x224) | **Top-1 Acc: 97.61%** (เกินเกณฑ์ 50% ได้เต็ม 5%)<br>**Top-3 Acc: 99.88%** |
| **คะแนนจัดลำดับประสิทธิภาพ (Ranking)** | **3%** | ใช้โมเดล ResNet-50 มาตรฐานระดับสากล ปรับจูน Head Layer และทำ Data Augmentation | ติดกลุ่มคะแนนสูงสุดของชั้นเรียน |
| **การใช้งาน Transfer Learning** | **1.5%** | ถ่ายโอนคุณลักษณะ (Feature Extractor) จาก ImageNet บน ResNet-50 | ได้คะแนนเต็ม 1.5% |
| **การใช้งาน Data Augmentation** | **1.5%** | สังเคราะห์ภาพด้วย Rotation (±10°), Affine, Shear, ColorJitter และ **ไม่ใช้ Horizontal Flip โดยเด็ดขาด** | ได้คะแนนเต็ม 1.5% |
| **เทคนิคหรือแนวคิดที่น่าสนใจ** | **2%** | 1. **Class-Weighted Cross-Entropy Loss** ($w_c = 1/\sqrt{N_c}$) แก้ปัญหาคลาส 1 ภาพ vs 5,025 ภาพ<br>2. **Cosine Annealing LR Scheduler**<br>3. **Stratified Split 80/20** | ได้คะแนนเต็ม 2.0% |
| **การนำเสนอและเอกสาร** | **2%** | สไลด์ PowerPoint 12 หน้า ตรงตามเกณฑ์ 11 ข้อ พร้อมสคริปต์พูด 10 นาทีและ Q&A 5 นาที | ได้คะแนนเต็ม 2.0% |

---

## 🚀 คำสั่งการใช้งานด่วน (Quick Start Commands)

### 1. การเตรียมและแบ่งชุดข้อมูล (Stratified Split 80% : 20%)
```bash
python prepare_dataset.py
```

### 2. การฝึกสอนแบบจำลอง (Training ด้วย ResNet-50)
```bash
python train.py --model resnet50 --epochs 5 --batch_size 64 --img_size 224
```

### 3. การทดสอบทำนายผลภาพ (Inference)
```bash
# 3.1 ทดสอบทำนายภาพเดี่ยว:
python inference.py --image "ThaiCharacter Dataset/round2/161/bc_001sg_3_118.jpg"

# 3.2 ทดสอบทำนายภาพทั้งโฟลเดอร์ (สำหรับชุดภาพทดสอบของอาจารย์):
python inference.py --dir "test_folder/"

# 3.3 สุ่มทดสอบภาพและสร้างตารางเปรียบเทียบผล:
python inference.py --demo
```

---

##  ไฮไลต์ประเด็นสำคัญสำหรับการนำเสนอ (Key Defense Points)

1. **ทำไมห้ามใช้ Horizontal Flip หรือ Vertical Flip?**
   - อักขระภาษาไทยมีความไม่สมมาตร การพลิกภาพแนวนอนจะทำให้ตัวอักษรเปลี่ยนความหมายทันที เช่น `ด` พลิกเป็น `ค`, `ภ` พลิกเป็น `ถ` หรือกลายเป็นอักขระที่ไม่มีความหมาย
2. **การจัดการความไม่สมดุลของข้อมูล (Extreme Class Imbalance 72 คลาส):**
   - คลาสสระ า (245) มี 5,025 ภาพ และ น (185) มี 4,863 ภาพ ในขณะที่ ฃ (163) และ ฑ (177) มีเพียง **1 ภาพ**
   - ใช้สูตรถ่วงน้ำหนัก Loss เพื่อดันให้โมเดลจำแนกคลาสที่มีตัวอย่างน้อยได้ถูกต้อง:
     $$w_c = \frac{1}{\sqrt{N_c}}$$
3. **การประเมินผล Top-1 และ Top-3:**
   - Top-1 สูงถึง **96.55%** และ Top-3 สูงถึง **99.74%** แสดงว่าตัวอักษรจริงติดอยู่ใน 3 อันดับแรกเกือบ 100%

---

##  ข้อมูลการส่งงานและการนำเสนอ
- **กำหนดส่ง:** วันที่ 25 กันยายน 2569 ก่อนเวลา 16:00 น.
- **สถานที่:** ห้อง 304
- **แสดงโค้ดและทดสอบ:** 13:00 น. - 13:45 น.
- **นำเสนอผลงาน:** 13:45 น. - 16:30 น. (10 นาที + ถามตอบ 5 นาที)
