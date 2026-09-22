# โครงการพัฒนาแบบจำลอง CNN สำหรับการรู้จำตัวอักษรและตัวเลขภาษาไทย (72 คลาส)
## Thai Character Recognition using CNN, Transfer Learning & Data Augmentation

### โครงงานนี้นำเสนอการพัฒนาแบบจำลอง Convolutional Neural Network (CNN) สถาปัตยกรรม ResNet-18 ร่วมกับเทคนิค Transfer Learning, Data Augmentation, Class-Weighted Cross-Entropy Loss และ Resolution Optimization (64 x 64) สำหรับจำแนกตัวอักษร สระ วรรณยุกต์ และตัวเลขภาษาไทยจำนวน 72 คลาส (ชุดข้อมูลรวม 62,709 ภาพ)

---

## ผลการทดลองและประสิทธิภาพแบบจำลอง (Experiment Results จาก model.pt)

แบบจำลองได้รับการฝึกสอนด้วยความละเอียดภาพที่เหมาะสมที่สุด 64 x 64 พิกเซล บนฮาร์ดแวร์ NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4, Mixed Precision FP16) สถาปัตยกรรม ResNet-18 (บันทึกค่าน้ำหนักที่ดีที่สุดที่ Epoch 14):

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Top-1 Acc (%) | Val Top-3 Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0.4433 | 91.72% | 0.1306 | 96.84% | 99.74% |
| 02 | 0.1508 | 96.15% | 0.1473 | 96.71% | 99.82% |
| 03 | 0.1206 | 96.57% | 0.1299 | 95.37% | 99.84% |
| 04 | 0.1109 | 96.64% | 0.1086 | 96.26% | 99.85% |
| 05 | 0.0957 | 97.03% | 0.0784 | 97.57% | 99.90% |
| 06 | 0.0865 | 97.18% | 0.0780 | 97.66% | 99.90% |
| 07 | 0.0835 | 97.16% | 0.0814 | 97.39% | 99.94% |
| 08 | 0.0774 | 97.29% | 0.0842 | 96.98% | 99.93% |
| 09 | 0.0696 | 97.54% | 0.0851 | 96.99% | 99.93% |
| 10 | 0.0665 | 97.62% | 0.1052 | 96.35% | 99.91% |
| 11 | 0.0667 | 97.59% | 0.0734 | 97.66% | 99.92% |
| 12 | 0.0604 | 97.78% | 0.0724 | 97.68% | 99.92% |
| 13 | 0.0556 | 97.83% | 0.0874 | 97.13% | 99.90% |
| **14 (Best)** | **0.0508** | **97.91%** | **0.0663** | **97.87%** | **99.94%** |

> **สรุปผลลัพธ์สำคัญ:**
> - **Validation Top-1 Accuracy: 97.87%** (สามารถระบุตัวอักษรได้ถูกต้องเป็นอันดับ 1 สูงเกือบ 98%)
> - **Validation Top-3 Accuracy: 99.94%** (คำตอบจริงติดอยู่ใน Top-3 แทบ 100%)
> - **Validation Loss: 0.0663** (ไม่มีปัญหา Overfitting)
> - **Synthetic Test Set (432 ภาพ, 72 คลาส):** ทำความแม่นยำได้ **90.05% (389 / 432 ภาพ)**

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
Deep/
|-- README.md                 # เอกสารคู่มือและรายงานผลโครงงานฉบับย่อ
|-- presentation_summary.md   # เอกสารสรุปเนื้อหาสำหรับนำเสนอสไลด์ 11 หัวข้อครบถ้วน
|-- train.ipynb               # ฝึกสอนโมเดล ResNet-18 (สแกนโฟลเดอร์ตรง 80/20 ใน RAM)
|-- infer.ipynb               # โหลด model.pt ทำนายภาพเดี่ยว / ตาราง test.csv / test.xlsx
|-- infer_teacher.ipynb       # ทำนายตารางเฉพาะของอาจารย์ (Path, Image Name, Extension, Label, Result)
|-- model.pt                  # ไฟล์ค่าน้ำหนักที่ดีที่สุด (ResNet-18 64x64, Val Acc: 97.87%)
|-- out.csv                   # ไฟล์ผลลัพธ์การทำนายพร้อมส่งอาจารย์
|-- training_curves.png       # กราฟแสดงประวัติ Loss และ Accuracy ตลอด 20 Epochs
|-- evaluation_results.csv    # ผลการประเมินความแม่นยำรายภาพของ Synthetic Test Set
|-- test/                     # โฟลเดอร์ชุดข้อมูลทดสอบและไฟล์ test.csv
|-- synthetic_test_set/       # ชุดข้อมูลภาพสังเคราะห์ 432 ภาพ สำหรับทดสอบ Generalization
`-- ThaiCharacter Dataset/    # โฟลเดอร์ชุดข้อมูลภาพหลัก (72 คลาส รวม 62,709 ภาพ)
```

---

## สรุปการตอบโจทย์ตามเกณฑ์การให้คะแนน

| เกณฑ์การให้คะแนน | แนวทางการดำเนินงานและเทคนิคในโครงงานนี้ |
| :--- | :--- |
| **1. ประสิทธิภาพการทำนาย (Accuracy)** | ทำ Validation Top-1 ได้ **97.87%**, Top-3 **99.94%**, Synthetic Test **90.05%** |
| **2. การใช้งาน Transfer Learning** | ใช้ ResNet-18 Pre-trained ImageNet ถ่ายโอน Feature Extractor ระดับลึก |
| **3. การใช้งาน Data Augmentation** | RandomRotation (+-12 deg), RandomAffine, ColorJitter และ White Background Padding |
| **4. เทคนิคหรือแนวคิดที่น่าสนใจ** | 1. **Resolution Optimization (64x64):** ลดอาการเบลอของเส้นอักษร ดัน Acc Synthetic สู่ 90.05%<br>2. **Class-Weighted Loss:** สูตร w_c = 1/sqrt(N_c) ป้องกันคลาสหายากถูกละเลย<br>3. **Zero-Dependency Native TIS-620:** ถอดรหัสตัวอักษรไทยโดยตรง<br>4. **Universal Inference Pipeline:** รองรับไฟล์ภาพ PNG, JPG, GIF และตาราง CSV, XLSX อัตโนมัติ |
| **5. การนำเสนอและเอกสาร** | [presentation_summary.md](presentation_summary.md) |

---

## ขั้นตอนการทำงานของสมุดโค้ด (Notebook Workflow)

### 1. การฝึกสอนโมเดล (Training)
เปิดไฟล์ [train.ipynb](train.ipynb) แล้วกด **Run All**:
- สแกนโฟลเดอร์รูปภาพ `ThaiCharacter Dataset`
- แบ่ง Stratified Split 80% Train : 20% Val ใน RAM
- ฝึกสอนโมเดล **ResNet-18 (64 x 64)** ด้วย Cosine Annealing Scheduler + Class-Weighted Loss
- บันทึกโมเดลที่ดีที่สุดลงใน `model.pt` (บันทึกอัตโนมัติที่ Epoch 14)

### 2. การทดสอบและประเมินผล (Inference)
มีสมุดโค้ดสำหรับทำนาย 2 รูปแบบตามโครงสร้างตารางของอาจารย์:

- **รูปแบบที่ 1: ตารางทั่วไป หรือมีคอลัมน์ id** -> เปิดไฟล์ [infer.ipynb](infer.ipynb)
  - รองรับไฟล์ตารางทั้ง `test.csv`, `test.xlsx`, และ `test.xls`
  - รองรับภาพทุกนามสกุล (.png, .jpg, .gif, .webp, .bmp) พร้อมจัดการพื้นหลังโปร่งใส (Alpha/Transparency)
  - ส่งออกผลลัพธ์เป็น `out.csv` ในรูปแบบ `id,label`

- **รูปแบบที่ 2: ตารางเฉพาะของอาจารย์** -> เปิดไฟล์ [infer_teacher.ipynb](infer_teacher.ipynb)
  - รองรับตารางที่มีคอลัมน์: `Path | Image Name | Extension | Label | Result`
  - ประกอบพาธของรูปภาพจากโฟลเดอร์ย่อยอัตโนมัติ
  - เติมคำตอบลงในช่อง `Result` และคำนวณ Accuracy อัตโนมัติหากมีเฉลยในช่อง `Label`
  - บันทึกตารางเดิมที่เติมคำตอบแล้วออกเป็น `out.csv`
