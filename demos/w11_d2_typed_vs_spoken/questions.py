# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 的題目：40 題四選一（常識 14、算術 13、兩步推理 13）＋ 10 題帶格式限制的。每題有英語與華語兩個版本。

這些題目是為這門課寫的，不取自任何現成的評測集（兩個理由：授權；現成的題目模型可能在訓練時看過）。
題目刻意簡單——要量的不是模型會不會，而是「同一題換一種方式進去，答案變不變」。
正解的位置在四個選項之間是平均的（各 10 題）；`./present.sh selftest` 會檢查。

同一題在四個條件下用的是**同一個字串**（`prompt()` 的輸出）：打字的條件直接送這個字串，
說的條件把同一個字串交給 TTS 念。華語版的選項代號也是 A–D（2026-11-27 使用者決定：選項一般就是用英文字母；原本用「甲乙丙丁」，Whisper small 把它們聽成「假以餅釘」，cascade 那一列 12 題只對 3 題）。
"""
from __future__ import annotations

LABELS = {"en": ["A", "B", "C", "D"], "zh": ["A", "B", "C", "D"]}

# (id, 類別, 正解的位置 0–3, 英語題幹, 英語選項, 華語題幹, 華語選項)
MC = [
    # ── 常識（14）
    ("k01", "knowledge", 1, "Which planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"],
     "哪一顆行星離太陽最近？", ["金星", "水星", "火星", "地球"]),
    ("k02", "knowledge", 2, "How many days are there in a leap year?", ["364", "365", "366", "367"],
     "閏年有幾天？", ["364 天", "365 天", "366 天", "367 天"]),
    ("k03", "knowledge", 3, "Which gas do plants take in from the air to make their food?", ["oxygen", "nitrogen", "hydrogen", "carbon dioxide"],
     "植物行光合作用的時候，從空氣中吸收哪一種氣體？", ["氧氣", "氮氣", "氫氣", "二氧化碳"]),
    ("k04", "knowledge", 0, "Which is the largest ocean on Earth?", ["the Pacific Ocean", "the Atlantic Ocean", "the Indian Ocean", "the Arctic Ocean"],
     "地球上最大的海洋是哪一個？", ["太平洋", "大西洋", "印度洋", "北極海"]),
    ("k05", "knowledge", 1, "At sea level, at what temperature in degrees Celsius does water boil?", ["90", "100", "110", "120"],
     "在海平面的高度，水在攝氏幾度沸騰？", ["90 度", "100 度", "110 度", "120 度"]),
    ("k06", "knowledge", 3, "Which of these animals is a mammal?", ["a shark", "a penguin", "a crocodile", "a dolphin"],
     "下面哪一種動物是哺乳類？", ["鯊魚", "企鵝", "鱷魚", "海豚"]),
    ("k07", "knowledge", 0, "How many sides does a hexagon have?", ["six", "five", "seven", "eight"],
     "六邊形有幾個邊？", ["六個", "五個", "七個", "八個"]),
    ("k08", "knowledge", 2, "Which organ pumps blood through the human body?", ["the lungs", "the liver", "the heart", "the kidneys"],
     "人體裡負責把血液送到全身的是哪一個器官？", ["肺", "肝", "心臟", "腎臟"]),
    ("k09", "knowledge", 0, "What is the capital of Japan?", ["Tokyo", "Kyoto", "Osaka", "Nagoya"],
     "日本的首都是哪裡？", ["東京", "京都", "大阪", "名古屋"]),
    ("k10", "knowledge", 3, "Which metal is a liquid at room temperature?", ["iron", "copper", "aluminium", "mercury"],
     "哪一種金屬在室溫下是液體？", ["鐵", "銅", "鋁", "水銀"]),
    ("k11", "knowledge", 1, "In which direction does the Sun rise?", ["the north", "the east", "the south", "the west"],
     "太陽從哪一個方向升起？", ["北方", "東方", "南方", "西方"]),
    ("k12", "knowledge", 2, "What is the smallest prime number?", ["zero", "one", "two", "three"],
     "最小的質數是多少？", ["零", "一", "二", "三"]),
    ("k13", "knowledge", 3, "Which language has the most native speakers in the world?", ["English", "Spanish", "Hindi", "Mandarin Chinese"],
     "世界上以哪一種語言為母語的人最多？", ["英語", "西班牙語", "印地語", "華語"]),
    ("k14", "knowledge", 0, "Which of these mountains is the highest above sea level?", ["Mount Everest", "Mount Fuji", "Mount Kilimanjaro", "Mont Blanc"],
     "下面哪一座山的海拔最高？", ["聖母峰", "富士山", "吉力馬札羅山", "白朗峰"]),
    # ── 算術（13）
    ("a01", "arithmetic", 2, "What is 17 plus 26?", ["41", "42", "43", "44"],
     "17 加 26 等於多少？", ["41", "42", "43", "44"]),
    ("a02", "arithmetic", 1, "What is 9 times 7?", ["56", "63", "72", "61"],
     "9 乘以 7 等於多少？", ["56", "63", "72", "61"]),
    ("a03", "arithmetic", 0, "What is 100 minus 37?", ["63", "67", "73", "57"],
     "100 減 37 等於多少？", ["63", "67", "73", "57"]),
    ("a04", "arithmetic", 3, "What is 144 divided by 12?", ["10", "11", "14", "12"],
     "144 除以 12 等於多少？", ["10", "11", "14", "12"]),
    ("a05", "arithmetic", 1, "What is half of 250?", ["115", "125", "120", "150"],
     "250 的一半是多少？", ["115", "125", "120", "150"]),
    ("a06", "arithmetic", 3, "What is 15 percent of 200?", ["15", "20", "25", "30"],
     "200 的百分之十五是多少？", ["15", "20", "25", "30"]),
    ("a07", "arithmetic", 0, "What is 3 squared plus 4 squared?", ["25", "7", "12", "49"],
     "3 的平方加 4 的平方等於多少？", ["25", "7", "12", "49"]),
    ("a08", "arithmetic", 2, "What is 12 times 12?", ["124", "122", "144", "148"],
     "12 乘以 12 等於多少？", ["124", "122", "144", "148"]),
    ("a09", "arithmetic", 1, "What is 25 times 4?", ["90", "100", "110", "125"],
     "25 乘以 4 等於多少？", ["90", "100", "110", "125"]),
    ("a10", "arithmetic", 3, "How many minutes are there in two and a half hours?", ["120", "130", "180", "150"],
     "兩個半小時是幾分鐘？", ["120 分鐘", "130 分鐘", "180 分鐘", "150 分鐘"]),
    ("a11", "arithmetic", 0, "What is 81 divided by 9?", ["9", "7", "8", "11"],
     "81 除以 9 等於多少？", ["9", "7", "8", "11"]),
    ("a12", "arithmetic", 2, "What is 0.5 plus 0.25?", ["0.55", "0.7", "0.75", "0.8"],
     "0.5 加 0.25 等於多少？", ["0.55", "0.7", "0.75", "0.8"]),
    ("a13", "arithmetic", 1, "What is 2 to the power of 5?", ["10", "32", "16", "25"],
     "2 的 5 次方是多少？", ["10", "32", "16", "25"]),
    # ── 兩步推理（13）
    ("r01", "reasoning", 1, "Tom has 3 apples. He buys 5 more, and then gives 2 away. How many apples does he have now?", ["5", "6", "7", "8"],
     "小明有 3 顆蘋果，又買了 5 顆，然後送給別人 2 顆。他現在有幾顆蘋果？", ["5 顆", "6 顆", "7 顆", "8 顆"]),
    ("r02", "reasoning", 2, "A train leaves at three in the afternoon, and the trip takes two and a half hours. When does it arrive?",
     ["at five o'clock", "at six o'clock", "at half past five", "at half past six"],
     "一班火車下午三點出發，車程兩個半小時。它幾點到？", ["五點", "六點", "五點半", "六點半"]),
    ("r03", "reasoning", 3, "A book costs 80 dollars, and today it is 25 percent off. How much do you pay?", ["55 dollars", "65 dollars", "70 dollars", "60 dollars"],
     "一本書定價 80 元，今天打七五折。要付多少錢？", ["55 元", "65 元", "70 元", "60 元"]),
    ("r04", "reasoning", 2, "Mary is older than John, and John is older than Sue. Who is the youngest?", ["Mary", "John", "Sue", "it cannot be determined"],
     "小美比小華大，小華比小玲大。誰的年紀最小？", ["小美", "小華", "小玲", "無法判斷"]),
    ("r05", "reasoning", 0, "Today is Wednesday. What day of the week will it be ten days from now?", ["Saturday", "Friday", "Sunday", "Monday"],
     "今天是星期三。十天之後是星期幾？", ["星期六", "星期五", "星期日", "星期一"]),
    ("r06", "reasoning", 3, "A rectangle is 4 metres wide and 6 metres long. What is its perimeter?", ["10 metres", "24 metres", "48 metres", "20 metres"],
     "一個長方形寬 4 公尺、長 6 公尺。它的周長是多少？", ["10 公尺", "24 公尺", "48 公尺", "20 公尺"]),
    ("r07", "reasoning", 1, "Each box holds 6 eggs. You need to pack 20 eggs. What is the smallest number of boxes you need?", ["3", "4", "5", "6"],
     "一個盒子可以裝 6 顆蛋。要裝 20 顆蛋，最少需要幾個盒子？", ["3 個", "4 個", "5 個", "6 個"]),
    ("r08", "reasoning", 0, "I have 50 dollars. I buy 3 notebooks that cost 12 dollars each. How much money do I have left?", ["14 dollars", "12 dollars", "16 dollars", "26 dollars"],
     "我有 50 元，買了 3 本筆記本，每本 12 元。我還剩多少錢？", ["14 元", "12 元", "16 元", "26 元"]),
    ("r09", "reasoning", 2, "A car travels at 60 kilometres per hour for one and a half hours. How far does it go?",
     ["80 kilometres", "100 kilometres", "90 kilometres", "120 kilometres"],
     "一輛車以時速 60 公里開了一個半小時。它開了多遠？", ["80 公里", "100 公里", "90 公里", "120 公里"]),
    ("r10", "reasoning", 0, "All cats are animals. Mimi is a cat. Which of these must be true?",
     ["Mimi is an animal", "all animals are cats", "Mimi is a dog", "Mimi is not an animal"],
     "所有的貓都是動物。咪咪是一隻貓。下面哪一句一定是對的？", ["咪咪是動物", "所有的動物都是貓", "咪咪是一隻狗", "咪咪不是動物"]),
    ("r11", "reasoning", 3, "A class has 30 students, and 40 percent of them are absent today. How many students are present?", ["12", "20", "24", "18"],
     "一個班有 30 個學生，今天有百分之四十的人請假。今天有幾個人到？", ["12 個", "20 個", "24 個", "18 個"]),
    ("r12", "reasoning", 1, "Yesterday was Friday. What day of the week is tomorrow?", ["Saturday", "Sunday", "Monday", "Thursday"],
     "昨天是星期五。明天是星期幾？", ["星期六", "星期日", "星期一", "星期四"]),
    ("r13", "reasoning", 2, "Three friends share two pizzas equally. Each pizza is cut into 6 slices. How many slices does each friend get?", ["2", "3", "4", "6"],
     "三個朋友平分兩個披薩，每個披薩切成 6 片。每個人分到幾片？", ["2 片", "3 片", "4 片", "6 片"]),
]

# 帶格式限制的 10 題：限制附在題目後面（Speech-IFEval 的做法）。rule 是 score.format_kept() 的規則；
# key 是內容的正解（可以沒有）——格式有沒有守住與答案對不對分開記。
#   letter       只有一個選項的代號（A–D）
#   number       只有一個數字（阿拉伯數字，或拼出來的數字）
#   yes_no       只有 yes／no（是／不是／對／不對）
#   max_units n  英語：不超過 n 個詞；華語：不超過 n 個字
#   units n      英語：正好 n 個詞；華語：正好 n 個字
#   items n      正好 n 項，用逗號（頓號）分開
#   prefix s     以 s 開頭
FORMAT = [
    ("f01", {"rule": "letter"}, 2,
     "Which of these is a fruit?", ["a carrot", "a potato", "a banana", "an onion"], "Reply with the letter only.",
     "下面哪一個是水果？", ["紅蘿蔔", "馬鈴薯", "香蕉", "洋蔥"], "只回答一個字母：A、B、C 或 D。"),
    ("f02", {"rule": "letter"}, 1,
     "Which of these numbers is even?", ["3", "8", "11", "15"], "Reply with the letter only.",
     "下面哪一個數字是偶數？", ["3", "8", "11", "15"], "只回答一個字母：A、B、C 或 D。"),
    ("f03", {"rule": "number", "key": {"en": ["13", "thirteen"], "zh": ["13", "十三"]}}, None,
     "What is 8 plus 5?", None, "Reply with the number only.",
     "8 加 5 等於多少？", None, "只回答數字。"),
    ("f04", {"rule": "number", "key": {"en": ["8", "eight"], "zh": ["8", "八"]}}, None,
     "How many legs does a spider have?", None, "Reply with the number only.",
     "蜘蛛有幾隻腳？", None, "只回答數字。"),
    ("f05", {"rule": "yes_no", "key": {"en": ["yes"], "zh": ["是", "對"]}}, None,
     "Is the Earth larger than the Moon?", None, "Answer yes or no, and nothing else.",
     "地球比月球大嗎？", None, "只回答「是」或「不是」。"),
    ("f06", {"rule": "yes_no", "key": {"en": ["no"], "zh": ["不是", "不對", "不"]}}, None,
     "Is ice hotter than steam?", None, "Answer yes or no, and nothing else.",
     "冰比水蒸氣熱嗎？", None, "只回答「是」或「不是」。"),
    ("f07", {"rule": "max_units", "n": {"en": 3, "zh": 5}, "key": {"en": ["green"], "zh": ["綠"]}}, None,
     "What colour do you get when you mix blue and yellow?", None, "Answer in at most three words.",
     "藍色加黃色會變成什麼顏色？", None, "用不超過五個字回答。"),
    ("f08", {"rule": "units", "n": {"en": 1, "zh": 1}, "key": {"en": ["cold"], "zh": ["冷"]}}, None,
     "What is the opposite of hot?", None, "Answer with one word only.",
     "「熱」的相反是什麼？", None, "只用一個字回答。"),
    ("f09", {"rule": "items", "n": 3}, None,
     "Name three days of the week.", None, "Separate them with commas, and say nothing else.",
     "說出一個星期裡的三天。", None, "用頓號分開，不要說其他的話。"),
    ("f10", {"rule": "prefix", "s": {"en": "answer:", "zh": "答案："}, "key": {"en": ["paris"], "zh": ["巴黎"]}}, None,
     "What is the capital of France?", None, "Begin your reply with the word Answer, followed by a colon.",
     "法國的首都是哪裡？", None, "回答的開頭請用「答案：」。"),
]

CLOSER = {"en": "Which one is correct: A, B, C, or D?", "zh": "請問答案是 A、B、C、D 的哪一個？"}


def _options(lang: str, opts: list[str]) -> str:
    lab = LABELS[lang]
    if lang == "zh":
        return " ".join(f"{lab[i]}，{o}。" for i, o in enumerate(opts))
    return " ".join(f"{lab[i]}: {o}." for i, o in enumerate(opts))


def items() -> list[dict]:
    """全部 50 題，每題：id、kind（knowledge／arithmetic／reasoning／format）、answer（0–3 或 None）、
    options[lang]、prompt[lang]（四個條件共用的那個字串）、fmt（格式題的規則）。"""
    out = []
    for qid, kind, ans, en_q, en_o, zh_q, zh_o in MC:
        out.append({"id": qid, "kind": kind, "answer": ans, "options": {"en": en_o, "zh": zh_o}, "fmt": None,
                    "prompt": {"en": f"{en_q} {_options('en', en_o)} {CLOSER['en']}",
                               "zh": f"{zh_q}{_options('zh', zh_o)}{CLOSER['zh']}"}})
    for qid, fmt, ans, en_q, en_o, en_c, zh_q, zh_o, zh_c in FORMAT:
        en = f"{en_q} {_options('en', en_o)} {en_c}" if en_o else f"{en_q} {en_c}"
        zh = f"{zh_q}{_options('zh', zh_o)}{zh_c}" if zh_o else f"{zh_q}{zh_c}"
        out.append({"id": qid, "kind": "format", "answer": ans, "options": {"en": en_o, "zh": zh_o} if en_o else None,
                    "fmt": fmt, "prompt": {"en": en, "zh": zh}})
    return out


def by_id() -> dict:
    return {q["id"]: q for q in items()}


if __name__ == "__main__":
    for q in items():
        print(q["id"], q["kind"], "|", q["prompt"]["en"])
        print(" " * len(q["id"]), " " * len(q["kind"]), "|", q["prompt"]["zh"])
