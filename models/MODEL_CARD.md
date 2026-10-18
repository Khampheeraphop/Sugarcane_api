# อ้อยดี: Model card

สถานะ: **experimental** ไม่ใช่ระบบวินิจฉัยพร้อมใช้ในไร่จริง

- MobileNetV3 Small (ImageNet) freeze backbone, embedding 576 มิติ + disease head 5 กลุ่ม และ sugarcane gate แยกกัน
- เทรน CPU Ryzen 5 4600H บนโน้ตบุ๊กนี้ ใช้ 4 threads; คำนวณรวมสกัด feature และเทรนประมาณ 54.8 วินาที ไม่รวมดาวน์โหลด/ติดตั้ง
- อ้อย 638 ภาพ SugarcaneLD-BD; negatives ที่ใช้ 1,000 ภาพดอกไม้และ 120 synthetic controls
- แยก pHash groups ก่อน train/validation/calibration/test; ไม่มีรหัสต้น/ไร่จึงไม่อ้างว่าเป็นการทดสอบข้ามแปลง

## Current application policy (support-v2)

Raw model weights are unchanged. A separate policy uses class-specific embedding-support cutoffs from the calibration split only. The policy JSON is bound to the weights by SHA-256. Missing or mismatched policy fails closed. The app does not present classifier confidence as a disease probability.

On the same held-out test split, 56/98 sugarcane images receive a supported pattern/healthy-pattern result; 55/56 match the source label. The other 42/98 are withheld as uncertain. None of 162 tested negatives receive a named disease. See `policy_metrics.json`. These are small curated dataset results, not Thai farm accuracy.

All leaf-only RedRot predictions require confirmation using other plant parts and are withheld, even with high scores. A user-reported out-of-scope leaf previously received a high-scoring RedRot result. It now yields uncertain; that private feedback image was used only as a regression case, never as training or calibration data. This does not add a trained leaf-pale disease class or establish its diagnosis.

## Historical raw classifier evaluation

The following scores describe the original classifier before the current support policy. They must not be presented as current application coverage or real-farm performance.

ผล held-out test ที่ไม่ได้ใช้เลือกเกณฑ์:

| กลุ่ม | จำนวนภาพ | Recall |
|---|---:|---:|
| EyeSpot | 12 | 12/12 |
| Healthy | 32 | 32/32 |
| RedLeafSpot | 7 | **4/7** |
| RedRot | 34 | 34/34 |
| RingSpot | 13 | 11/13 |

Accuracy รวมก่อนปฏิเสธ = **93/98 = 94.9%**; Macro-F1 = **0.910**

เมื่อใช้ gate/threshold ให้ผลได้ 97/98 ภาพอ้อย และถูกต้อง 93/97 = 95.9% ภาพนอกกลุ่มที่ทดสอบ 162 ภาพผ่าน gate ผิด 0 ภาพ **ไม่ได้หมายความว่าจะปฏิเสธภาพทุกชนิดได้ 100%** เพราะ negatives ไม่ครอบคลุมหญ้า ข้าว ข้าวโพด สิ่งของและโรคที่ไม่รู้จัก

จุดอ่อนชัดเจนคือ RedLeafSpot: ในชุดทดสอบมีเพียง 7 ภาพและพลาด 3 ภาพ จึงไม่ควรใช้คะแนนรวมเป็นเหตุผลให้นำไปใช้ตัดสินใจในแปลง ไม่มีข้อมูลภาคสนามไทยหรือโรคระยะแฝง ไม่มีโมเดลความเสี่ยงอากาศ

คะแนนหลัง temperature scaling ไม่ใช่โอกาสยืนยันเชื้อหรือสาเหตุจริงของต้นนั้น คำแนะนำโรคเป็นฐานความรู้ที่อ้างอิงคู่มือภายนอก ไม่ได้พิสูจน์ด้วยโมเดล ความสัมพันธ์ระหว่าง label ภาพกับชื่อเชื้อยังต้องตรวจโดยผู้เชี่ยวชาญ

โหมดสำรวจแปลงใช้การเปรียบเทียบสี RGB/HSV กับพื้นที่อ้างอิงที่ผู้ใช้เลือก ไม่ใช้โมเดลนี้ ไม่ระบุโรค และไม่ใช่โมเดลที่เทรนจากภาพโดรน

ดูหลักฐานรายภาพใน `test_predictions.json` รายละเอียดคะแนนใน `metrics.json` และที่มาของข้อมูล/สิทธิ์ใช้งานใน README.md กับ `provenance/sources.json`; per-image test predictions are generated locally and not committed
