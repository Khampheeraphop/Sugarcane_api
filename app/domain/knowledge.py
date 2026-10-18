"""Reviewed general disease information. Never inferred causes of an individual case.
Source labels are mapped to common disease names, not proof of the pathogen.
"""

SRA = "https://sugarresearch.com.au/wp-content/uploads/2025/05/Diseases-of-Australian-Sugarcane_Fieldguide_2022_web.pdf"
TNAU = "https://agritech.tnau.ac.in/expert_system/sugar/cropprotection.html"
FAQ = "https://agritech.tnau.ac.in/expert_system/sugar/faq.html"

KNOWLEDGE = {
    "EyeSpot": {
        "cause": "โรคใบจุดตาโดยทั่วไปเกี่ยวข้องกับเชื้อรา Bipolaris sacchari แต่ภาพนี้ยังไม่ยืนยันเชื้อ",
        "favorable_conditions": "สภาพอากาศเย็นและชื้นเอื้อต่อโรค ต้องตรวจสภาพแปลงจริงประกอบ",
        "check": ["ตรวจจุดรูปรีและรอยยาวบนใบ ถ่ายทั้งสองด้านของใบ"],
        "care": ["ติดตามการลุกลามและปรึกษาเจ้าหน้าที่เรื่องพันธุ์ต้านทานที่เหมาะกับพื้นที่"],
        "sources": [{"title": "Sugar Research Australia: Eye spot (หน้า 23)", "url": SRA}],
    },
    "RedLeafSpot": {
        "cause": "โรคใบจุดแดงตามคู่มือ SRA เกี่ยวข้องกับเชื้อรา Dimeriella sacchari อาการแดงจากสาเหตุอื่นอาจคล้ายกัน",
        "favorable_conditions": "อากาศอบอุ่นและชื้นเอื้อต่อโรค ไม่ได้แปลว่าความชื้นเป็นสาเหตุเดียว",
        "check": ["ตรวจผิวใบด้านบนและใบส่วนบนของทรงพุ่มเพิ่มเติม"],
        "care": ["บันทึกอาการและสอบถามเจ้าหน้าที่เรื่องพันธุ์ต้านทาน หากอาการเพิ่มขึ้นควรส่งตัวอย่างตรวจ"],
        "sources": [{"title": "Sugar Research Australia: Red leaf spot (หน้า 25)", "url": SRA}],
    },
    "RingSpot": {
        "cause": "โรคใบจุดวงแหวนโดยทั่วไปเกี่ยวข้องกับเชื้อรา Leptosphaeria sacchari ภาพอย่างเดียวไม่ยืนยันเชื้อ",
        "favorable_conditions": "อากาศอบอุ่นและชื้นเอื้อต่อโรค ต้องตรวจอายุใบและสภาพแปลงร่วมด้วย ระบบยังไม่ได้วัดอากาศของแปลงนี้",
        "check": ["ถ่ายแผลที่มีใจกลางซีดและขอบเข้ม รวมทั้งภาพใบเก่าและทั้งต้น"],
        "care": ["ติดตามอาการบนใบใหม่ และให้เจ้าหน้าที่ประเมินก่อนตัดสินใจจัดการแปลง"],
        "sources": [{"title": "Sugar Research Australia: Ring spot (หน้า 26)", "url": SRA}],
    },
    "RedRot": {
        "cause": "โรคเน่าแดงมีสาเหตุจากเชื้อรา Colletotrichum falcatum การติดเชื้ออาจมากับท่อนพันธุ์ที่เป็นโรค",
        "favorable_conditions": "น้ำฝน น้ำชลประทาน และลมอาจช่วยแพร่เชื้อ แต่ระบบยังระบุไม่ได้ว่าต้นนี้ได้รับเชื้อทางใด",
        "check": [
            "ต้องตรวจลำต้นประกอบ โดยให้ผู้เชี่ยวชาญตรวจเนื้อในที่แดงสลับปื้นขาว ภาพใบอย่างเดียวไม่พอยืนยันโรคเน่าแดง"
        ],
        "care": [
            "งดนำต้นที่สงสัยไปใช้เป็นท่อนพันธุ์จนกว่าจะตรวจยืนยัน",
            "ให้เจ้าหน้าที่เกษตรตรวจแปลงและแนะนำท่อนพันธุ์สะอาดหรือพันธุ์ต้านทานที่เหมาะกับพื้นที่",
        ],
        "sources": [
            {"title": "TNAU: Red rot symptoms and management", "url": TNAU},
            {"title": "TNAU: Red rot spread and typical symptoms", "url": FAQ},
        ],
    },
    "Healthy": {
        "cause": "โมเดลยังไม่พบกลุ่มอาการที่รู้จักจากภาพนี้ ไม่ได้ตรวจเชื้อหรือความผิดปกติทุกชนิด",
        "favorable_conditions": "ยังไม่มีข้อมูลอากาศและผลสำรวจแปลง จึงไม่ประเมินความเสี่ยงในอนาคต",
        "check": ["สังเกตใบใหม่ ยอด และลำต้นร่วมด้วย"],
        "care": ["ถ่ายติดตามต้นเดิม หากเริ่มมีอาการให้ตรวจใหม่หรือปรึกษาเจ้าหน้าที่"],
        "sources": [],
    },
}


def advice(class_id):
    return {
        **KNOWLEDGE[class_id],
        "scope": "ข้อมูลโรคโดยทั่วไป ไม่ใช่การยืนยันสาเหตุของต้นที่ถ่ายภาพ",
        "treatment_note": "ยังไม่แนะนำสารเคมีหรืออัตราใช้จากภาพคัดกรอง ควรยืนยันโรคและตรวจคำแนะนำที่ขึ้นทะเบียนในพื้นที่ก่อน",
    }
