# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 2：Biasing——提詞卡的三種用法（同一段含專有名詞的音訊，名單從三個不同的層進去）。

  show       課堂：載入 Whisper small、wav2vec2-base-960h、Qwen3-ASR，終端機單鍵操作（下面的鍵）。離開時存 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，照課堂順序全部跑一遍。英文的結果存 runs/rehearsal/（進版控 = 現場的退路）；
             華語的結果存 ../_media/w07_d2/private_rehearsal.*（不進版控、不上站——裡面有自錄句的文字與名單）
  replay     現場模型出狀況時：從兩份彩排紀錄讀回，同樣的鍵、同樣的畫面，不碰 torch
  check      課前體檢：版本、三個模型在不在快取、Qwen3-ASR 在 MPS 還是 CPU 跑得起來、prompt 槽通不通、prefix beam δ=0 對 greedy → runs/check.json
  pick       從 toml 的英文候選句裡挑課堂用的那一句：每句過 Whisper 無 prompt 與 CTC greedy，印出哪些名字錯了
  fetch      課前一週：下載模型（Whisper、wav2vec2 的 commit 讀兄弟 demo 的 model.lock；Qwen3-ASR 取最新 commit）、解出英文候選句的 wav；要網路
  record     錄一句自己的華語（含三四個專有名詞）→ ../_media/w07_d2/mandarin.wav，接著輸入參考文字與名單 → mandarin.txt、names.txt
  fake       沒有模型也能演練流程（罐頭輸出，畫面標明 FAKE，不是給學生看的）
  selftest   不需要模型：trie 與 failure arc、prefix beam search 對窮舉、名字命中、繁簡判定、洩漏判定、fake 整條流程
  export     runs/rehearsal/ 與 private_rehearsal.json 的摘要 → slides/assets/w07/w07-metrics.json（鍵 demo2；華語只進「幾個名字對、字體、有沒有洩漏」，不進文字）

畫面上的鍵（順序照投影片 p50 講稿的【操作】）：
  1  英文那句 → Whisper，prompt 空            2  同一句，prompt = 名單
  e  同一句，prompt = 一句台詞（英文版的 6）
  3  華語那句 → Whisper，prompt 空            4  prompt = 名單（繁體）
  5  prompt = 一句繁體寫的話（沒有名字）        6  prompt = 一句台詞
  7  英文那句 → wav2vec2 的 logits：greedy、prefix beam δ = 0、δ 三個值（名單進 trie）——每一條標出名字進來了沒、哪些詞被改掉
  8  Qwen3-ASR：英文與華語各兩次（context 空／名單）
  9  Qwen3-ASR：華語，context = 繁體那句／台詞那句（與 5、6 並排）
  0  沒有語音時提詞卡會怎樣：8 s 近乎靜音的合成音訊（極低的白噪音）配台詞 prompt → Whisper（en、zh）與 Qwen3-ASR（en、zh）——Demo 1 的「證據歸零」接到這裡
  p  印出全部 prompt 的原文      t  投影片 p50 的表      h  重印選單      q  離開

判定的定義（課堂上要念出來的）：
  名字命中   英文：轉錄小寫、去標點後，名字當一個詞串出現；華語：兩邊都轉成簡體再找子字串（字體不算錯）
  字體       華語輸出裡「只在簡體用的字」對「只在繁體用的字」的數目（內建一張常用對照表，不是 OpenCC）
  洩漏       prompt 裡連續 3 個英文詞／4 個華語字出現在轉錄裡
  改掉的詞   δ > 0 的輸出對 δ = 0 的輸出做詞級 diff，列出被換掉的詞——這就是「δ 太大，同音的常用詞變成名字」要看的東西
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from ctc_prefix_beam import HotwordBonus, Trie, greedy as ctc_greedy, prefix_beam_search  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
DIG = int(CFG["display"]["digits"])
MODEL_LOCK = HERE / "model.lock"
PRIVATE = (HERE / CFG["mandarin"]["dir"]).resolve()
METRICS = (HERE.parent.parent / "slides" / "assets" / "w07" / "w07-metrics.json").resolve()
SIBLING_LOCKS = {CFG["whisper"]["repo"]: [HERE.parent / "w07_d1_hallucination" / "model.lock", HERE.parent / "w04_d3_attention_heads" / "model.lock"],
                 CFG["ctc"]["repo"]: [HERE.parent / "w07_d1_hallucination" / "model.lock", HERE.parent / "w03_d3_blank_heatmap" / "model.lock"]}


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── 純函數（selftest 測這些）──────────────────────────────────────────────
# 繁簡對照（常用字；只收兩邊寫法不同的）。這不是 OpenCC，只拿來判定「這段輸出偏繁還是偏簡」與讓名字比對不受字體影響。
_S2T_PAIRS = (
    "国國 们們 这這 个個 为為 时時 会會 来來 说說 对對 开開 发發 当當 后後 么麼 过過 还還 没沒 样樣 经經 现現 让讓 问問 题題 关關 东東 车車 门門 长長 "
    "间間 电電 话話 书書 学學 见見 气氣 几幾 两兩 点點 边邊 里裡 头頭 机機 业業 产產 买買 卖賣 钱錢 儿兒 认認 识識 记記 语語 请請 谢謝 师師 医醫 爱愛 "
    "应應 该該 从從 动動 听聽 写寫 读讀 给給 进進 远遠 运運 带帶 导導 单單 办辦 场場 实實 验驗 阳陽 阴陰 风風 云雲 飞飛 马馬 鸟鳥 鱼魚 龙龍 龟龜 乐樂 "
    "药藥 园園 区區 县縣 镇鎮 乡鄉 号號 楼樓 层層 网網 线線 终終 结結 组組 级級 统統 续續 练練 细細 纪紀 约約 红紅 绿綠 蓝藍 黄黃 鸡雞 猪豬 鸭鴨 虾蝦 "
    "热熱 灾災 烟煙 灯燈 烧燒 补補 装裝 裤褲 衬襯 视視 观觀 览覽 觉覺 讨討 论論 议議 设設 计計 词詞 诉訴 译譯 试試 诗詩 诚誠 调調 谈談 谁誰 课課 谱譜 "
    "贝貝 负負 贵貴 贸貿 费費 资資 赛賽 赵趙 赶趕 转轉 轻輕 载載 输輸 辑輯 达達 迁遷 选選 递遞 适適 遗遺 邮郵 钟鐘 钢鋼 铁鐵 铜銅 银銀 错錯 锁鎖 镜鏡 "
    "闭閉 闲閒 阅閱 队隊 际際 险險 随隨 隐隱 难難 雾霧 页頁 顶頂 项項 顺順 须須 顾顧 预預 领領 频頻 颜顏 额額 飘飄 饭飯 饮飲 饺餃 馆館 驾駕 骑騎 骗騙 "
    "鲁魯 鲜鮮 鸣鳴 鹅鵝 麦麥 齐齊 齿齒 龄齡 台臺 湾灣 系係 体體 万萬 与與 专專 丑醜 严嚴 丰豐 临臨 众眾 优優 伙夥 众眾 传傳 伦倫 价價 侨僑 兰蘭 关關 "
    "兴興 军軍 农農 决決 况況 净淨 准準 凤鳳 划劃 则則 刘劉 创創 别別 剂劑 办辦 劲勁 势勢 华華 协協 卫衛 厅廳 历歷 压壓 厌厭 厉厲 县縣 参參 双雙 "
    "变變 叠疊 吓嚇 吗嗎 吴吳 启啟 员員 呐吶 响響 哟喲 唤喚 喂餵 嘱囑 团團 围圍 园園 图圖 圆圓 坏壞 块塊 坚堅 垒壘 执執 扩擴 扫掃 扬揚 护護 报報 "
    "担擔 拟擬 择擇 挂掛 挡擋 据據 损損 换換 摄攝 据據 数數 断斷 旧舊 显顯 晓曉 术術 权權 条條 极極 构構 标標 栏欄 树樹 样樣 检檢 楼樓 欢歡 "
    "欧歐 岁歲 归歸 杀殺 汇匯 汉漢 汤湯 沪滬 泪淚 浅淺 测測 济濟 浏瀏 涌湧 温溫 湿濕 满滿 滚滾 灭滅 灵靈 烦煩 烧燒 热熱 爷爺 狱獄 独獨 猎獵 "
    "玛瑪 环環 现現 琼瓊 疗療 疯瘋 盖蓋 盘盤 础礎 确確 础礎 祸禍 离離 种種 积積 称稱 稳穩 穷窮 窃竊 竞競 笃篃 笔筆 筑築 签簽 简簡 类類 粤粵 "
    "紧緊 累累 继繼 续續 维維 综綜 缓緩 编編 缘緣 罗羅 罚罰 罢罷 艺藝 节節 苏蘇 范範 荐薦 获獲 营營 蔑衊 虑慮 虚虛 虫蟲 虽雖 蚀蝕 蜡蠟 补補 "
    "袭襲 誉譽 认認 讯訊 讲講 讽諷 证證 评評 识識 诊診 译譯 详詳 诞誕 谅諒 谊誼 谋謀 谜謎 谨謹 谱譜 豬豬 账賬 贡貢 贤賢 败敗 货貨 质質 购購 "
    "贯貫 贴貼 贷貸 赋賦 赢贏 赶趕 趋趨 跃躍 践踐 踊踴 车車 轨軌 转轉 轮輪 软軟 较較 辅輔 辆輛 辈輩 辽遼 迹跡 适適 选選 逊遜 递遞 遥遙 邓鄧 "
    "郑鄭 酿釀 释釋 钥鑰 钻鑽 铃鈴 铅鉛 链鏈 销銷 锅鍋 锋鋒 锦錦 键鍵 镇鎮 镜鏡 长長 闪閃 闯闖 闹鬧 阎閻 阔闊 际際 陆陸 陈陳 隆隆 隶隸 难難 "
    "雏雛 雳靂 需需 静靜 韩韓 顿頓 颁頒 颂頌 颗顆 题題 颠顛 飙飆 饱飽 饼餅 馈饋 驰馳 驱驅 驻駐 骂罵 鬓鬢 魂魂 鱿魷 鳄鱷 鸿鴻 鹏鵬 麽麼 黏黏 "
    "齐齊 龚龔 诚誠 决決 够夠 够夠 产產 廣廣 广廣 厂廠 录錄 么麼 于於 后後 奖獎 妆妝 婶嬸 孙孫 导導 层層 屬屬 属屬 岛島 币幣 师師 带帶 帐帳 帮幫 庙廟 庄莊 "
    "庆慶 库庫 应應 废廢 异異 弃棄 张張 弹彈 强強 归歸 彻徹 径徑 态態 总總 恶惡 恳懇 悬懸 惧懼 惭慚 惩懲 惯慣 愿願 懂懂 戏戲 战戰 户戶 "
    "扑撲 托托 扰擾 抛拋 抢搶 拢攏 拥擁 挤擠 捣搗 掷擲 搂摟 摆擺 撑撐 擦擦 收收 效效 敌敵 斋齋 斩斬 断斷 旷曠 昼晝 晋晉 暂暫 曲曲 杰傑 "
    "松鬆 枢樞 柜櫃 栋棟 档檔 桥橋 梦夢 棂欞 椅椅 榄欖 楽樂 概概 槛檻 橱櫥 歼殲 殴毆 毁毀 毕畢 气氣 氢氫 氯氯 汹洶 沟溝 没沒 沈沈 "
    "泻瀉 洁潔 洒灑 浆漿 浑渾 浓濃 涂塗 涛濤 润潤 涨漲 渐漸 渔漁 渗滲 湾灣 溃潰 滞滯 漓灕 潇瀟 澜瀾 灿燦 炉爐 烂爛 炼煉 烛燭 焕煥 焖燜 "
    "煙煙 爱愛 牍牘 牵牽 犹猶 狈狽 狭狹 猫貓 献獻 玑璣 环環 珐琺 琏璉 瑶瑤 瓯甌 甦甦 疖癤 疟瘧 疮瘡 症癥 痒癢 痪瘓 痹痺 瘪癟 癞癩 皱皺 "
    "盏盞 盐鹽 监監 盗盜 盟盟 眯眯 睁睜 瞩矚 矫矯 硕碩 碍礙 碱鹼 磁磁 礼禮 祷禱 禅禪 秃禿 秆稈 税稅 稣穌 稻稻 穂穗 窍竅 窑窯 窜竄 窝窩 "
    "竖豎 笋筍 笼籠 筹籌 签簽 箩籮 篓簍 簖籪 籴糴 粜糶 粪糞 糊糊 系系 絷縶 纠糾 纤纖 纪紀 纬緯 纯純 纱紗 纲綱 纳納 纵縱 纷紛 纸紙 纹紋 "
    "纺紡 纽紐 线線 绀紺 练練 组組 绅紳 细細 织織 终終 绊絆 绍紹 经經 绑綁 绒絨 结結 绕繞 绘繪 给給 绚絢 络絡 绝絕 绞絞 统統 绢絹 绣繡 "
    "绥綏 绩績 绪緒 续續 绮綺 绯緋 绰綽 绳繩 维維 绵綿 绷繃 绸綢 综綜 绽綻 绿綠 缀綴 缄緘 缅緬 缆纜 缇緹 缉緝 缎緞 缓緩 缔締 缕縷 编編 "
    "缘緣 缚縛 缝縫 缠纏 缤繽 缨纓 缩縮 缭繚 缮繕 缰韁 缴繳 罂罌 网網 罚罰 罢罷 罴羆 羁羈 翘翹 耢耮 聂聶 聋聾 职職 聩聵 聪聰 肃肅 肠腸 "
    "肤膚 肮骯 肾腎 肿腫 胀脹 胁脅 胆膽 胜勝 胧朧 胶膠 脉脈 脍膾 脏髒 脐臍 脑腦 脓膿 脚腳 脱脫 脸臉 腊臘 腭齶 腻膩 腾騰 膑臏 臜臢 舆輿 "
    "舣艤 舰艦 舱艙 艳艷 艺藝 苁蓯 苇葦 苈藶 苋莧 苌萇 苍蒼 苎苧 苏蘇 茏蘢 茑蔦 茔塋 茕煢 荆荊 荚莢 荛蕘 荜蓽 荞蕎 荟薈 荠薺 荡蕩 荣榮 "
    "荤葷 荥滎 荦熒 荧熒 荨蕁 荩蓀 荪蓀 荫蔭 荬蕒 荭葒 荮葤 药藥 莱萊 莲蓮 莳蒔 莴萵 莸蕕 莹瑩 莺鶯 萝蘿 萤螢 营營 萦縈 萧蕭 萨薩 葱蔥 "
    "蒇蕆 蒉蕢 蒋蔣 蒌蔞 蓝藍 蓟薊 蓠蘺 蓣蕷 蔂虆 蔷薔 蔹蘝 蔺藺 蔼藹 蕲蘄 蕴蘊 薮藪 藓蘚 蘖蘗 虏虜 虑慮 虚虛 虬虯 虮蟣 虾蝦 虿蠆 蚀蝕 "
    "蚁蟻 蚂螞 蚕蠶 蚝蠔 蚬蜆 蛊蠱 蛎蠣 蛏蟶 蛮蠻 蛱蛺 蛲蟯 蛳螄 蛴蠐 蜕蛻 蜗蝸 蜡蠟 蝇蠅 蝈蟈 蝉蟬 蝼螻 蝾蠑 螀螿 螨蟎 蟏蠨 衅釁 衔銜 "
    "补補 衬襯 袄襖 袜襪 袭襲 袯襏 装裝 裆襠 裢褳 裣襝 裤褲 裥襇 褛褸 褴襤 襕襴 见見 观觀 觃覎 规規 觅覓 视視 觇覘 览覽 觉覺 觊覬 觋覡 "
    "觌覿 觍覥 觎覦 觏覯 觐覲 觑覷 觞觶 触觸 觯觶 誊謄 讠訁 计計 订訂 讣訃 认認 讥譏 讦訐 讧訌 讨討 让讓 讪訕 讫訖 训訓 议議 讯訊 记記 "
    "讲講 讳諱 讴謳 讵詎 讶訝 讷訥 许許 讹訛 论論 讻訩 讼訟 讽諷 设設 访訪 诀訣 证證 诂詁 诃訶 评評 诅詛 识識 诇詗 诈詐 诉訴 诊診 诋詆 "
    "诌謅 词詞 诎詘 诏詔 诐詖 译譯 诒詒 诓誆 诔誄 试試 诖詿 诗詩 诘詰 诙詼 诚誠 诛誅 诜詵 话話 诞誕 诟詬 诠詮 诡詭 询詢 诣詣 诤諍 该該 "
    "详詳 诧詫 诨諢 诩詡 诪譸 诫誡 诬誣 语語 诮誚 误誤 诰誥 诱誘 诲誨 诳誑 说說 诵誦 诶誒 请請 诸諸 诹諏 诺諾 读讀 诼諑 诽誹 课課 诿諉 "
    "谀諛 谁誰 谂諗 调調 谄諂 谅諒 谆諄 谇誶 谈談 谊誼 谋謀 谌諶 谍諜 谎謊 谏諫 谐諧 谑謔 谒謁 谓謂 谔諤 谕諭 谖諼 谗讒 谘諮 谙諳 谚諺 "
    "谛諦 谜謎 谝諞 谞諝 谟謨 谠讜 谡謖 谢謝 谣謠 谤謗 谥謚 谦謙 谧謐 谨謹 谩謾 谪謫 谫譾 谬謬 谭譚 谮譖 谯譙 谰讕 谱譜 谲譎 谳讞 谴譴 "
    "谵譫 谶讖 貝貝 贞貞 负負 贠貟 贡貢 财財 责責 贤賢 败敗 账賬 货貨 质質 贩販 贪貪 贫貧 贬貶 购購 贮貯 贯貫 贰貳 贱賤 贲賁 贳貰 贴貼 "
    "贵貴 贶貺 贷貸 贸貿 费費 贺賀 贻貽 贼賊 贽贄 贾賈 贿賄 赀貲 赁賃 赂賂 赃贓 资資 赅賅 赆贐 赇賕 赈賑 赉賚 赊賒 赋賦 赌賭 赍賫 赎贖 "
    "赏賞 赐賜 赑贔 赒賙 赓賡 赔賠 赕賧 赖賴 赗賵 赘贅 赙賻 赚賺 赛賽 赜賾 赝贗 赞贊 赟贇 赠贈 赡贍 赢贏 赣贛"
)


def _build_maps() -> tuple[dict[str, str], dict[str, str]]:
    s2t, t2s = {}, {}
    for pair in _S2T_PAIRS.split():
        if len(pair) != 2 or pair[0] == pair[1]:
            continue
        s, t = pair
        s2t.setdefault(s, t)
        t2s.setdefault(t, s)
    return s2t, t2s


S2T, T2S = _build_maps()
# 這些「簡體字」在繁體文裡也常見（台北、系統、范姓、松樹……），只拿來對名字、不拿來判字體
AMBIG = set("台系划松后范蔑丑杰")


def to_simplified(text: str) -> str:
    return "".join(T2S.get(ch, ch) for ch in text)


def script_judge(text: str) -> dict:
    """{"simp": 只在簡體用的字數, "trad": 只在繁體用的字數, "verdict": "繁" | "簡" | "混" | "無法判定"}"""
    simp = sum(1 for ch in text if ch in S2T and ch not in T2S and ch not in AMBIG)
    trad = sum(1 for ch in text if ch in T2S and ch not in S2T)
    if simp == 0 and trad == 0:
        v = "無法判定"
    elif simp and trad:
        v = "混"
    else:
        v = "繁" if trad else "簡"
    return {"simp": simp, "trad": trad, "verdict": v}


def norm_en(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def name_hits(text: str, names: list[str], zh: bool) -> dict[str, bool]:
    """英文：小寫、去標點後名字當一個詞串出現（詞界）；華語：兩邊轉簡體後找子字串。"""
    out = {}
    if zh:
        t = to_simplified(text)
        for n in names:
            out[n] = to_simplified(n) in t
    else:
        t = " " + norm_en(text) + " "
        for n in names:
            out[n] = (" " + norm_en(n) + " ") in t
    return out


def leak(prompt: str, text: str, zh: bool, n: int) -> list[str]:
    """prompt 裡連續 n 個詞（英文）或 n 個字（華語）出現在轉錄裡的片段。"""
    found = []
    if zh:
        units, t = list(to_simplified(re.sub(r"[\s，。、！？；：「」]", "", prompt))), to_simplified(text)
        join, wrap = (lambda u: "".join(u)), (lambda s: s)
    else:
        units, t = norm_en(prompt).split(), " " + norm_en(text) + " "
        join, wrap = (lambda u: " ".join(u)), (lambda s: " " + s + " ")
    i = 0
    while i + n <= len(units):                       # 從 i 起最長的、整段出現在轉錄裡的片段
        j = i + n
        if wrap(join(units[i:j])) not in t:
            i += 1
            continue
        while j < len(units) and wrap(join(units[i:j + 1])) in t:
            j += 1
        found.append(join(units[i:j]))
        i = j
    return found


def word_diff(base: str, other: str) -> dict:
    """詞級 diff：base 裡被換掉／刪掉的詞 → other 裡換成什麼。"""
    a, b = norm_en(base).split(), norm_en(other).split()
    changed = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            changed.append({"from": " ".join(a[i1:i2]), "to": " ".join(b[j1:j2])})
    return {"n_changed": len(changed), "changes": changed}


def wer_words(ref: str, hyp: str) -> dict:
    r, h = norm_en(ref).split(), norm_en(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return {"errors": int(d[-1, -1]), "n_ref": len(r), "wer": float(d[-1, -1] / len(r)) if r else 0.0}


def fmt_list(names: list[str], zh: bool) -> str:
    return ("、" if zh else ", ").join(names)


# ── 模型 ─────────────────────────────────────────────────────────────────
def read_lock() -> dict[str, str]:
    lock = {}
    if MODEL_LOCK.exists():
        for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                lock[parts[0]] = parts[1]
    for repo, sibs in SIBLING_LOCKS.items():
        if repo in lock:
            continue
        for sib in sibs:
            if sib.exists():
                for line in sib.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == repo:
                        lock[repo] = parts[1]
                    elif len(parts) == 1 and parts[0]:          # W3 d3、W4 d3 的 lock 是單行 sha
                        lock[repo] = parts[0]
                if repo in lock:
                    break
    return lock


PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.jinja", "*.model", "*.tiktoken"]


def model_dir(repo: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline, allow_patterns=PATTERNS)


def quiet_transformers() -> None:
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def pick_device(name: str):
    import torch
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def strip_prefix(seq: list[int], sot: int, eot: int, n_prefix: int) -> list[int]:
    if sot in seq:
        seq = seq[seq.index(sot) + n_prefix:]
    else:
        while seq and seq[0] >= eot:
            seq = seq[1:]
    if eot in seq:
        seq = seq[: seq.index(eot)]
    return [int(t) for t in seq if int(t) < eot]


class Whisper:
    """transformers 的 WhisperForConditionalGeneration：generate（prompt_ids 可選）＋ teacher-forced 的平均 log 機率。"""
    name = "whisper"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        quiet_transformers()
        self.torch = torch
        w = CFG["whisper"]
        path = model_dir(w["repo"])
        self.proc = WhisperProcessor.from_pretrained(path)
        self.tok = self.proc.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(path).eval()
        self.device = pick_device(device or w["device"])
        self.model.to(self.device)
        gen, cfg = self.model.generation_config, self.model.config
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.sot = int(gen.decoder_start_token_id)
        self.eot = int(gen.eos_token_id if gen.eos_token_id is not None else cfg.eos_token_id)
        self.notimestamps = int(gen.no_timestamps_token_id)
        self.lang_to_id, self.task_to_id = gen.lang_to_id, gen.task_to_id
        self.task = w["task"]

    def prefix(self, language: str) -> list[int]:
        return [self.sot, int(self.lang_to_id[f"<|{language}|>"]), int(self.task_to_id[self.task]), self.notimestamps]

    def prompt_ids(self, text: str | None) -> list[int]:
        if not text:
            return []
        return [int(t) for t in np.asarray(self.proc.get_prompt_ids(text, return_tensors="np")).tolist()]

    def features(self, x: np.ndarray):
        return self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt", return_attention_mask=True)

    def transcribe(self, x: np.ndarray, language: str, prompt: str | None) -> dict:
        torch = self.torch
        w = CFG["whisper"]
        inp = self.features(x)
        kw = dict(language=language, task=self.task, return_timestamps=False, max_new_tokens=int(w["max_new_tokens"]),
                  do_sample=False, num_beams=int(w["num_beams"]), return_dict_in_generate=True)
        pid = self.prompt_ids(prompt)
        if pid:
            kw["prompt_ids"] = torch.tensor(pid, dtype=torch.long, device=self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(inp.input_features.to(self.device), attention_mask=inp.attention_mask.to(self.device), **kw)
        dt = time.perf_counter() - t0
        seqs = out.sequences if hasattr(out, "sequences") else out
        ids = strip_prefix([int(t) for t in seqs[0].tolist()], self.sot, self.eot, len(self.prefix(language)))
        # teacher-forced 一次前向：平均 log 機率（Demo 1 的定義，含 <|endoftext|>）
        pre = pid + self.prefix(language)
        full = pre + ids + [self.eot]
        dec = torch.tensor([full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            lg = self.model(input_features=inp.input_features.to(self.device), decoder_input_ids=dec).logits[0].float()
        logp = torch.log_softmax(lg, dim=-1).cpu().numpy()
        lps = [float(logp[j, full[j + 1]]) for j in range(len(pre) - 1, len(full) - 1)]
        return {"text": self.tok.decode(ids, skip_special_tokens=True).strip(), "n_tokens": len(ids), "n_prompt_tokens": len(pid),
                "avg_logprob": float(sum(lps) / len(lps)) if lps else 0.0, "seconds": dt}


class CTC:
    """transformers 的 Wav2Vec2ForCTC（W3 Demo 3 的模型）：只拿 log softmax，解碼都在 ctc_prefix_beam.py。"""
    name = "wav2vec2"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        quiet_transformers()
        self.torch = torch
        path = model_dir(CFG["ctc"]["repo"])
        self.proc = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).eval()
        self.device = pick_device(device or CFG["ctc"]["device"])
        self.model.to(self.device)
        self.blank = int(self.model.config.pad_token_id)
        self.vocab = {k: int(v) for k, v in self.proc.tokenizer.get_vocab().items()}
        self.delim = self.vocab.get(self.proc.tokenizer.word_delimiter_token, self.vocab.get("|"))
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def logprobs(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        inp = self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inp.input_values.to(self.device)).logits[0].float()
        return torch.log_softmax(logits, dim=-1).cpu().numpy()

    def encode(self, phrase: str) -> tuple[int, ...]:
        """名單上的片語 → 字元 id（大寫；空白 → 詞界符號）。不在符號表裡的字元丟掉（例如重音字母）。"""
        ids = []
        for ch in phrase.upper().strip():
            if ch == " ":
                ids.append(self.delim)
            elif ch in self.vocab:
                ids.append(self.vocab[ch])
        return tuple(ids)

    def decode(self, ids: list[int]) -> str:
        return self.proc.tokenizer.decode(ids, group_tokens=False)


class Qwen:
    """Qwen3-ASR 的 transformers 原生版（≥ 5.13.0）：apply_transcription_request(audio, language, prompt) → generate → decode。
    context 走 prompt 參數（model card：「free-form context … via prompt to bias the transcription」，與 chat template 的 system message 是同一個槽）。"""
    name = "qwen3-asr"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoProcessor
        quiet_transformers()
        self.torch = torch
        q = CFG["qwen"]
        path = model_dir(q["repo"])
        self.proc = AutoProcessor.from_pretrained(path)
        try:
            from transformers import Qwen3ASRForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForMultimodalLM as Cls
        self.device = pick_device(device or q["device"])
        dtype = getattr(torch, q["dtype"]) if self.device.type != "cpu" else torch.float32
        self.model = Cls.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def _inputs(self, audio, language: str | None, context: str | None):
        kw = {"audio": audio}
        if language:
            kw["language"] = language
        if context:
            kw["prompt"] = context
        return self.proc.apply_transcription_request(**kw)

    def transcribe(self, wav_path: Path, x: np.ndarray, language: str | None, context: str | None) -> dict:
        torch = self.torch
        try:
            inputs = self._inputs(str(wav_path), language, context)
            how = "path"
        except Exception:
            inputs = self._inputs(x.astype(np.float32), language, context)      # 讀檔那條不通就直接給陣列
            how = "array"
        inputs = inputs.to(self.device, self.dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=int(CFG["qwen"]["max_new_tokens"]), do_sample=False)
        dt = time.perf_counter() - t0
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self.proc.decode(gen)
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        try:
            text = self.proc.decode(gen, return_format="transcription_only")
            text = text[0] if isinstance(text, (list, tuple)) else text
        except Exception:
            text = raw.split("<asr_text>", 1)[-1] if "<asr_text>" in raw else raw
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", str(text)).strip()
        return {"text": text, "raw": str(raw).strip(), "n_tokens": int(gen.shape[1]), "seconds": dt, "audio_as": how}


# ── fake 後端（沒有模型時演練流程）────────────────────────────────────────────
class FakeWhisper:
    name = "fake"
    device = "fake"
    n_params = 0

    def transcribe(self, x, language, prompt):
        if x is not None and len(x) and float(np.max(np.abs(x))) < 0.01:      # 近乎靜音：把卡念出來（2026-11-01 真模型就是這樣）
            text = prompt or "you"
            return {"text": text, "n_tokens": len(text.split()), "n_prompt_tokens": len((prompt or "").split()), "avg_logprob": -0.19, "seconds": 0.0}
        if language == "zh":
            text = "我们今天要去台北的实验室找王老师。"
            if prompt and script_judge(prompt)["verdict"] == "繁":
                text = "我們今天要去臺北的實驗室找王老師。"
            if prompt and "謝謝大家" in prompt:
                text = "謝謝大家，下次見。我們今天要去臺北的實驗室找王老師。"
        else:
            text = "Think of Lady Glenarvin, think of Mary Grant."
            if prompt and "Glenarvan" in prompt:
                text = "Think of Lady Glenarvan, think of Mary Grant."
            if prompt and "subscribe" in prompt:
                text = "Don't forget to subscribe to the channel. Think of Lady Glenarvin, think of Mary Grant."
        return {"text": text, "n_tokens": len(text.split()), "n_prompt_tokens": len((prompt or "").split()), "avg_logprob": -0.31, "seconds": 0.0}


class FakeCTC:
    name = "fake"
    device = "fake"
    blank, delim = 0, 4
    # 'a'..'z' → 5..30，'|' → 4，"'" → 31；與 wav2vec2-base-960h 的符號表同一個順序不重要，selftest 只看邏輯
    vocab = {"|": 4, "'": 31, **{chr(ord("A") + i): 5 + i for i in range(26)}}
    inv = {v: k for k, v in vocab.items()}

    def encode(self, phrase):
        return tuple(self.vocab.get(ch, self.vocab["|"]) if ch != " " else 4 for ch in phrase.upper() if ch == " " or ch in self.vocab)

    def decode(self, ids):
        return "".join(self.inv.get(i, "") for i in ids).replace("|", " ").strip()

    def logprobs(self, x):
        """合成一段 log 機率：把 'THINK OF LADY GLENARVIN THINK OF MARY GRANT' 的字元每個攤兩格、中間夾 blank，
        GLENARVIN 的 I 那格把 A 也給一點機率——δ 夠大時 beam 會改成 GLENARVAN。"""
        text = "THINK OF LADY GLENARVIN THINK OF MARY GRANT"
        V = 32
        rows = []
        for ch in text:
            c = self.encode(ch)[0] if ch != " " else 4
            p = np.full(V, 1e-4)
            p[c] = 0.9
            if ch == "I" and len(rows) > 30 and len(rows) < 48:     # GLENARVIN 的那個 I
                p[c] = 0.55
                p[self.vocab["A"]] = 0.4
            p /= p.sum()
            rows.append(np.log(p))
            b = np.full(V, 1e-4)
            b[0] = 0.95
            b /= b.sum()
            rows.append(np.log(b))
        return np.array(rows)


class FakeQwen:
    name = "fake"
    device = "fake"
    n_params = 0

    def transcribe(self, wav_path, x, language, context):
        if x is not None and len(x) and float(np.max(np.abs(x))) < 0.01:
            text = context or "嗯。"
            return {"text": text, "raw": text + "<|im_end|>", "n_tokens": len(text), "seconds": 0.0, "audio_as": "fake"}
        if language == "Chinese":
            text = "我们今天要去台北的实验室找王老师。"
            if context and script_judge(context)["verdict"] == "繁":
                text = "我們今天要去臺北的實驗室找王老師。"
        else:
            text = "Think of Lady Glenarvan, think of Mary Grant." if context and "Glenarvan" in context else "Think of Lady Glen Arvin, think of Mary Grant."
        return {"text": text, "raw": "language English<asr_text>" + text, "n_tokens": len(text), "seconds": 0.0, "audio_as": "fake"}


# ── 輸出紀錄：公開與私密兩個水槽 ───────────────────────────────────────────
class Out:
    def __init__(self):
        self.pub: list[str] = []
        self.priv: list[str] = []
        self.private = False

    def __call__(self, s: str = "") -> None:
        print(s)
        (self.priv if self.private else self.pub).append(s)

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        (self.priv if self.private else self.pub).extend(["", "─" * 72, s, "─" * 72])

    def text(self, private: bool = False) -> str:
        return "\n".join(self.priv if private else self.pub) + "\n"


# ── 素材 ─────────────────────────────────────────────────────────────────
def english_candidate(utt: str | None = None) -> dict:
    utt = utt or CFG["english"]["utt"]
    for c in CFG["english"]["candidates"]:
        if c["utt"] == utt:
            return dict(c)
    sys.exit(f"toml 的 english.candidates 裡沒有 {utt}")


def english_wav_path(utt: str) -> Path:
    return _p(CFG["audio"]["w05_wav_dir"]) / f"{utt}.wav"


def extract_from_tar(utts: list[str]) -> list[str]:
    """從 dev-clean.tar.gz 解指定句子的 flac → 16 kHz wav（W5 Demo 1 沒解到的才需要）。要 soundfile。"""
    import tarfile
    tar = _p(CFG["audio"]["dev_clean_tar"])
    if not tar.exists():
        return []
    import soundfile as sf
    want = {u: f"{u}.flac" for u in utts}
    done = []
    with tarfile.open(tar, "r:gz") as tf:
        for m in tf:
            base = Path(m.name).name
            for u, fn in want.items():
                if base == fn:
                    f = tf.extractfile(m)
                    x, fs_in = sf.read(f, dtype="float64")
                    import dsp
                    x = dsp.resample(dsp.to_mono_float(x), fs_in, FS)
                    C.save_wav(english_wav_path(u), x, FS)
                    done.append(u)
    return done


def load_english(utt: str | None = None) -> tuple[dict, Path, np.ndarray]:
    c = english_candidate(utt)
    p = english_wav_path(c["utt"])
    if not p.exists():
        sys.exit(f"沒有 {p}，先跑 ./present.sh fetch（會從 dev-clean.tar.gz 解出來）")
    return c, p, C.load_wav(p, FS)


def quiet_clip() -> np.ndarray:
    """8 s 近乎靜音：極低的白噪音（RMS 在 toml），確定性。2026-11-01 彩排時誤錄到一段沒說話的室內底噪，兩個模型配 prompt 都把提詞卡念了出來——
    這段合成的是它可重現的版本（不進版控也不必：隨時可重做）。"""
    a = CFG["audio"]
    rng = np.random.default_rng(int(a["quiet_seed"]))
    x = rng.standard_normal(int(a["quiet_s"] * FS))
    return x / np.sqrt(np.mean(x ** 2)) * float(a["quiet_rms"])


def load_mandarin() -> tuple[dict, Path, np.ndarray] | None:
    wav, txt, names = PRIVATE / "mandarin.wav", PRIVATE / "mandarin.txt", PRIVATE / "names.txt"
    if not wav.exists():
        return None
    ref = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    nl = [l.strip() for l in names.read_text(encoding="utf-8").splitlines() if l.strip()] if names.exists() else []
    return {"text": ref, "names": nl}, wav, C.load_wav(wav, FS)


# ── Demo ────────────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, en: tuple | None, zh: tuple | None, whisper, ctc, qwen, stored: dict | None = None, stored_private: dict | None = None, tag: str = ""):
        self.out, self.wh, self.ctc, self.qw = out, whisper, ctc, qwen
        self.en, self.zh = en, zh
        self.res: dict = dict(stored or {})
        self.res_priv: dict = dict(stored_private or {})
        self.replay = stored is not None
        self.tag = tag
        P = CFG["prompts"]
        self.en_names = en[0]["names"] if en else ["Glenarvan", "Mary Grant"]
        self.zh_names = zh[0]["names"] if zh else []
        self.prompts_en = {"none": None, "list": P["list_fmt_en"].format(names=fmt_list(self.en_names, False)), "dialogue": P["dialogue_en"],
                           "qwen_list": P["qwen_list_fmt"].format(names=fmt_list(self.en_names, False))}
        self.prompts_zh = {"none": None, "list": P["list_fmt_zh"].format(names=fmt_list(self.zh_names, True)) if self.zh_names else None,
                           "traditional": P["traditional_zh"], "dialogue": P["dialogue_zh"],
                           "qwen_list": P["qwen_list_fmt"].format(names=fmt_list(self.zh_names, True)) if self.zh_names else None}

    # 快取（公開／私密分開存）
    def _get(self, key: str, compute, private: bool = False):
        store = self.res_priv if private else self.res
        if key in store:
            return store[key]
        if self.replay:
            self.out(f"  {C.AMBER}彩排紀錄裡沒有 {key}{C.RESET}")
            return None
        store[key] = compute()
        return store[key]

    # 一格：Whisper
    def _whisper_panel(self, which: str, kind: str) -> None:
        o = self.out
        zh = which == "zh"
        o.private = zh
        try:
            data = self.zh if zh else self.en
            if data is None:
                o("  沒有自錄的華語句：先 ./present.sh record" if zh else "  沒有英文那句：先 ./present.sh fetch")
                return
            ref, wav, x = data
            prompt = (self.prompts_zh if zh else self.prompts_en)[kind]
            if kind != "none" and prompt is None:
                o("  名單是空的：../_media/w07_d2/names.txt 一行一個名字"); return
            lang = CFG["whisper"]["language_zh" if zh else "language_en"]
            label = {"none": "prompt 空", "list": "prompt = 名單", "traditional": "prompt = 一句繁體寫的話（沒有名字）", "dialogue": "prompt = 一句台詞"}[kind]
            o.banner(f"{'華語自錄句' if zh else '英文 ' + ref['utt']} → Whisper small（<|{lang}|>），{label}")
            if prompt is not None:
                o(f"  prompt：{C.AMBER}{prompt!r}{C.RESET}")
            r = self._get(f"whisper:{which}:{kind}", lambda: self._whisper_run(x, lang, prompt, ref, zh), private=zh)
            if r:
                self._print_asr(r, ref, zh, prompt)
        finally:
            o.private = False

    def _whisper_run(self, x, lang, prompt, ref, zh) -> dict:
        r = self.wh.transcribe(x, lang, prompt)
        return self._annotate(r, ref, zh, prompt)

    def _annotate(self, r: dict, ref: dict, zh: bool, prompt: str | None) -> dict:
        names = ref["names"]
        r["hits"] = name_hits(r["text"], names, zh)
        r["n_hit"], r["n_names"] = int(sum(r["hits"].values())), len(names)
        r["script"] = script_judge(r["text"]) if zh else None
        d = CFG["display"]
        # 洩漏：prompt 的片段出現在轉錄裡。名單 prompt 的名字本來就該出現，先把名字從 prompt 裡拿掉再比（否則名單一命中就是「洩漏」）
        pr = prompt
        if pr:
            for n in names:
                pr = pr.replace(n, " ")
        r["leak"] = leak(pr, r["text"], zh, int(d["leak_ngram_zh" if zh else "leak_ngram_en"])) if pr and pr.strip(" ,.、。") else []
        if not zh and ref.get("text"):
            r["wer"] = wer_words(ref["text"], r["text"])
        return r

    def _print_asr(self, r: dict, ref: dict, zh: bool, prompt: str | None) -> None:
        o = self.out
        o(f"  轉錄：{C.RED}{r['text']!r}{C.RESET}" + (f"   ({r['seconds']:.2f} s)" if r.get("seconds") else ""))
        if ref.get("text"):
            o(f"  參考：{ref['text']!r}")
        hits = "、".join(f"{n} {'✓' if ok else '✗'}" for n, ok in r["hits"].items())
        extra = []
        if r.get("avg_logprob") is not None and "avg_logprob" in r:
            extra.append(f"平均 log 機率 {r['avg_logprob']:.{DIG}f}")
        if r.get("wer"):
            extra.append(f"詞錯 {r['wer']['errors']}/{r['wer']['n_ref']}")
        if r.get("script"):
            s = r["script"]
            extra.append(f"字體：{s['verdict']}（簡體專用字 {s['simp']}、繁體專用字 {s['trad']}）")
        o(f"  名字 {r['n_hit']}/{r['n_names']}：{hits}" + ("；" + "；".join(extra) if extra else ""))
        if prompt:
            o(f"  prompt 洩漏：{C.RED + '、'.join(repr(g) for g in r['leak']) + C.RESET if r['leak'] else '沒有'}")

    # 一格：CTC + hotwords
    def panel_ctc(self) -> None:
        o = self.out
        if self.en is None:
            o("  沒有英文那句：先 ./present.sh fetch"); return
        ref, wav, x = self.en
        cfg = CFG["ctc"]
        o.banner(f"英文 {ref['utt']} → wav2vec2-base-960h 的 log 機率 → greedy、prefix beam（{cfg['beam']} 條）、名單 trie + δ")
        o(f"  名單：{self.en_names}（字元級、只從詞界開始比對；走在路上每個字元 +δ，岔開扣回）")

        def compute():
            lp = self.ctc.logprobs(x)
            trie = Trie([self.ctc.encode(n) for n in self.en_names])
            rows = []
            g = ctc_greedy(lp, self.ctc.blank)
            rows.append({"delta": None, "label": "greedy", "text": self.ctc.decode(g), "logp": None, "bonus": 0.0})
            for d in [0.0] + list(cfg["deltas"]):
                t0 = time.perf_counter()
                best = prefix_beam_search(lp, self.ctc.blank, beam=int(cfg["beam"]), bonus=HotwordBonus(trie if d > 0 else None, d, self.ctc.delim),
                                          prune_logp=float(cfg["prune_logp"]), n_best=3)[0]
                rows.append({"delta": d, "label": f"beam δ={d:g}", "text": self.ctc.decode(best["ids"]), "logp": best["logp"], "bonus": best["bonus"],
                             "seconds": time.perf_counter() - t0})
            base = rows[1]["text"]
            for r in rows:
                r["hits"] = name_hits(r["text"], self.en_names, False)
                r["n_hit"] = int(sum(r["hits"].values()))
                r["wer"] = wer_words(ref["text"], r["text"])
                r["diff_vs_delta0"] = word_diff(base, r["text"]) if r["delta"] not in (None, 0.0) else None
            return {"T": int(lp.shape[0]), "rows": rows, "names": self.en_names, "trie_nodes": trie.n_nodes}
        r = self._get("ctc:en", compute)
        if not r:
            return
        o(f"  T = {r['T']} frame，trie {r['trie_nodes']} 個節點；參考：{ref['text']!r}")
        for row in r["rows"]:
            hits = "、".join(f"{n} {'✓' if ok else '✗'}" for n, ok in row["hits"].items())
            lp = f"log P {row['logp']:.{DIG}f} + bonus {row['bonus']:.1f}" if row["logp"] is not None else ""
            o(f"\n  {C.BOLD}{row['label']:<12s}{C.RESET} {C.RED}{row['text']!r}{C.RESET}")
            o(f"    名字 {row['n_hit']}/{len(r['names'])}：{hits}；詞錯 {row['wer']['errors']}/{row['wer']['n_ref']}" + (f"；{lp}" if lp else ""))
            if row.get("diff_vs_delta0") and row["diff_vs_delta0"]["n_changed"]:
                ch = "；".join(f"{c['from']!r} → {c['to']!r}" for c in row["diff_vs_delta0"]["changes"])
                o(f"    對 δ=0 改掉的：{C.AMBER}{ch}{C.RESET}")
            elif row.get("diff_vs_delta0"):
                o("    對 δ=0 沒有改動")
        o("\n  （分數層的 biasing 沒有「理解」：它只在名單的路上加分。δ 大到某個值，與名字聲學上接近的常用詞就會被改成名字——看上面「改掉的」那幾行）")

    # 一格：Qwen3-ASR
    def _qwen_panel(self, which: str, kinds: list[str]) -> None:
        o = self.out
        zh = which == "zh"
        o.private = zh
        try:
            data = self.zh if zh else self.en
            if data is None:
                o("  沒有自錄的華語句：先 ./present.sh record" if zh else "  沒有英文那句：先 ./present.sh fetch"); return
            if self.qw is None and not self.replay:
                o(f"  Qwen3-ASR 這台機器上載不起來（{getattr(self, 'qw_error', None)}）——只放 Whisper 與 CTC 兩格，理由課堂上說"); return
            ref, wav, x = data
            lang = CFG["qwen"]["language_zh" if zh else "language_en"]
            for kind in kinds:
                prompts = self.prompts_zh if zh else self.prompts_en
                ctx = prompts["qwen_list"] if kind == "list" else prompts[kind]
                if kind != "none" and ctx is None:
                    o("  名單是空的：../_media/w07_d2/names.txt"); continue
                label = {"none": "context 空", "list": "context = 名單", "traditional": "context = 一句繁體寫的話", "dialogue": "context = 一句台詞"}[kind]
                o.banner(f"{'華語自錄句' if zh else '英文 ' + ref['utt']} → Qwen3-ASR（{CFG['qwen']['repo'].split('/')[-1]}，language={lang}），{label}")
                if ctx:
                    o(f"  context：{C.AMBER}{ctx!r}{C.RESET}")
                r = self._get(f"qwen:{which}:{kind}", lambda k=kind, c=ctx: self._annotate(self.qw.transcribe(wav, x, lang, c), ref, zh, c), private=zh)
                if r:
                    self._print_asr(r, ref, zh, ctx)
                    if r.get("raw") and r["raw"] != r["text"]:
                        o(f"  原始輸出：{r['raw']!r}")
        finally:
            o.private = False

    # 沒有語音時的提詞卡
    def panel_quiet(self) -> None:
        o, P = self.out, CFG["prompts"]
        a = CFG["audio"]
        o.banner(f"沒有語音時提詞卡會怎樣：{a['quiet_s']:g} s 近乎靜音（白噪音 RMS {a['quiet_rms']:g}）配台詞 prompt → Whisper、Qwen3-ASR")
        x = quiet_clip()
        wav = HERE / "runs" / "quiet.wav"
        if not self.replay and not wav.exists():
            C.save_wav(wav, x, FS)
        ref = {"text": "", "names": []}
        d = CFG["display"]
        cases = [("whisper", "en", CFG["whisper"]["language_en"], P["dialogue_en"]), ("whisper", "zh", CFG["whisper"]["language_zh"], P["dialogue_zh"]),
                 ("whisper", "zh-trad", CFG["whisper"]["language_zh"], P["traditional_zh"]),
                 ("qwen", "en", CFG["qwen"]["language_en"], P["dialogue_en"]), ("qwen", "zh", CFG["qwen"]["language_zh"], P["dialogue_zh"]),
                 ("qwen", "zh-trad", CFG["qwen"]["language_zh"], P["traditional_zh"])]
        for model, lang_key, lang, prompt in cases:
            zh = lang_key.startswith("zh")
            if model == "qwen" and self.qw is None and not self.replay:
                o(f"  Qwen3-ASR 載不起來（{getattr(self, 'qw_error', None)}），跳過"); break
            o(f"\n  {C.BOLD}{'Whisper small' if model == 'whisper' else 'Qwen3-ASR'}，{lang}，prompt：{C.RESET}{C.AMBER}{prompt!r}{C.RESET}")

            def compute(m=model, l=lang, pr=prompt, z=zh):
                r = self.wh.transcribe(x, l, pr) if m == "whisper" else self.qw.transcribe(wav, x, l, pr)
                r["leak"] = leak(pr, r["text"], z, int(d["leak_ngram_zh" if z else "leak_ngram_en"]))
                r["echo_verbatim"] = re.sub(r"[\s，。、！？,.!?]", "", to_simplified(r["text"])) == re.sub(r"[\s，。、！？,.!?]", "", to_simplified(pr))
                return r
            r = self._get(f"quiet:{model}:{lang_key}", compute)
            if r:
                o(f"  轉錄：{C.RED}{r['text']!r}{C.RESET}" + (f"   ({r['seconds']:.2f} s)" if r.get("seconds") else "")
                  + (f"；平均 log 機率 {r['avg_logprob']:.{DIG}f}" if r.get("avg_logprob") is not None else ""))
                o(f"  prompt 洩漏：{C.RED + '、'.join(repr(g) for g in r['leak']) + C.RESET if r['leak'] else '沒有'}" + ("；**整張卡逐字念出**" if r["echo_verbatim"] else ""))
        o("\n  （證據是平的時候，先驗就是輸出——Demo 1 看到的是訓練資料的先驗，這裡看到的是你剛塞進 prompt 槽的那一句）")

    # 投影片的表
    def panel_table(self) -> None:
        o = self.out
        o.banner("投影片 p50 的表（名字對幾個／字體／洩漏）——華語那列只有數字，文字不上站")
        cols = [("Whisper 無 prompt", "whisper:{w}:none"), ("Whisper + 名單", "whisper:{w}:list"), ("Whisper + 繁體句", "whisper:{w}:traditional"),
                ("Whisper + 台詞", "whisper:{w}:dialogue"), ("Qwen3-ASR 無 context", "qwen:{w}:none"), ("Qwen3-ASR + 名單", "qwen:{w}:list")]

        def cell(store, key):
            r = store.get(key)
            if not r:
                return "—"
            s = f"{r['n_hit']}/{r['n_names']}"
            if r.get("script"):
                s += f" {r['script']['verdict']}"
            if r.get("leak"):
                s += " 洩漏"
            return s
        o(f"  {'':<10s}" + "".join(f"{c[0]:<22s}" for c in cols))
        o(f"  {'英文':<10s}" + "".join(f"{cell(self.res, c[1].format(w='en')):<22s}" for c in cols))
        o(f"  {'華語':<10s}" + "".join(f"{cell(self.res_priv, c[1].format(w='zh')):<22s}" for c in cols))
        c = self.res.get("ctc:en")
        if c:
            o("  CTC：" + "；".join(f"{r['label']} {r['n_hit']}/{len(c['names'])}" + (f"（改 {r['diff_vs_delta0']['n_changed']} 處）" if r.get("diff_vs_delta0") else "") for r in c["rows"]))
        e = self.res.get("whisper:en:dialogue")
        if e:
            o(f"  英文台詞 prompt 洩漏：{e['leak'] or '沒有'}")
        q = {k[6:]: v for k, v in self.res.items() if k.startswith("quiet:")}
        if q:
            o("  近乎靜音 + prompt：" + "；".join(f"{k} {'整張卡逐字' if v.get('echo_verbatim') else ('洩漏 ' + str(len(v['leak'])) + ' 段' if v['leak'] else '沒洩漏')}" for k, v in q.items()))

    def panel_prompts(self) -> None:
        o = self.out
        o.banner("全部 prompt／context 的原文")
        for k, v in self.prompts_en.items():
            o(f"  en {k:<12s} {v!r}")
        o.private = True
        for k, v in self.prompts_zh.items():
            o(f"  zh {k:<12s} {v!r}")
        o.private = False

    def menu(self) -> None:
        o = self.out
        o("\n  1 英文 Whisper 無 prompt   2 英文 + 名單   e 英文 + 台詞      3 華語 Whisper 無 prompt   4 華語 + 名單   5 華語 + 繁體句   6 華語 + 台詞")
        o("  7 英文 CTC：greedy／beam δ=0／δ 三個值   8 Qwen3-ASR 英文與華語（無／名單）   9 Qwen3-ASR 華語（繁體句／台詞）")
        o("  0 沒有語音時的提詞卡（近乎靜音 + 台詞 prompt → 兩個模型）   p 全部 prompt   t 表   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1": self._whisper_panel("en", "none")
        elif k == "2": self._whisper_panel("en", "list")
        elif k == "e": self._whisper_panel("en", "dialogue")
        elif k == "3": self._whisper_panel("zh", "none")
        elif k == "4": self._whisper_panel("zh", "list")
        elif k == "5": self._whisper_panel("zh", "traditional")
        elif k == "6": self._whisper_panel("zh", "dialogue")
        elif k == "7": self.panel_ctc()
        elif k == "8": self._qwen_panel("en", ["none", "list"]); self._qwen_panel("zh", ["none", "list"])
        elif k == "9": self._qwen_panel("zh", ["traditional", "dialogue"])
        elif k == "0": self.panel_quiet()
        elif k == "p": self.panel_prompts()
        elif k == "t": self.panel_table()
        elif k == "h": self.menu()
        elif k == "q": return False
        return True

    def run_all(self) -> None:
        for k in "12e34567":
            self.dispatch(k)
        self._qwen_panel("en", ["none", "list", "dialogue"])
        self._qwen_panel("zh", ["none", "list", "traditional", "dialogue"])
        self.panel_quiet()
        self.panel_table()

    def versions(self) -> dict:
        v = C.versions()
        from importlib import metadata
        for pkg in ("torch", "transformers", "huggingface_hub", "tokenizers"):
            try:
                v[pkg] = metadata.version(pkg)
            except Exception:
                v[pkg] = None
        v["models"] = read_lock()
        return v

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        dev = {"whisper": str(getattr(self.wh, "device", "")), "ctc": str(getattr(self.ctc, "device", "")), "qwen": str(getattr(self.qw, "device", "")) if self.qw else None}
        log = {"time": C.now(), "tag": self.tag, "backend": self.wh.name, "device": dev, "versions": self.versions(), "config": CFG,
               "english": {"utt": self.en[0]["utt"], "text": self.en[0]["text"], "names": self.en_names} if self.en else None,
               "prompts_en": self.prompts_en, "qwen_error": getattr(self, "qw_error", None),
               "mandarin": {"recorded": self.zh is not None, "n_names": len(self.zh_names)}, "results": self.res}
        (folder / "rehearsal.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / "rehearsal.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  英文的結果存在 {folder}/（rehearsal.json、rehearsal.txt）")
        if self.zh is not None or self.res_priv:
            PRIVATE.mkdir(parents=True, exist_ok=True)
            name = "private_rehearsal" if self.tag == "rehearsal" else f"private_{folder.name}"
            plog = {"time": log["time"], "tag": self.tag, "prompts_zh": self.prompts_zh, "mandarin": {"text": self.zh[0]["text"], "names": self.zh_names} if self.zh else None,
                    "results": self.res_priv}
            (PRIVATE / f"{name}.json").write_text(json.dumps(plog, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
            (PRIVATE / f"{name}.txt").write_text(self.out.text(private=True), encoding="utf-8")
            C.note(f"  華語的結果存在 {PRIVATE}/{name}.json、.txt（不進版控、不上站）")


# ── 建構 ─────────────────────────────────────────────────────────────────────
def build(cmd: str, out: Out) -> Demo:
    if cmd == "replay":
        src = HERE / "runs" / "rehearsal" / "rehearsal.json"
        if not src.exists():
            sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
        log = json.loads(src.read_text(encoding="utf-8"))
        psrc = PRIVATE / "private_rehearsal.json"
        plog = json.loads(psrc.read_text(encoding="utf-8")) if psrc.exists() else {}
        out.banner(f"REPLAY：{log['time']} 的彩排（backend {log['backend']}）" + ("" if plog else "；沒有華語那半的紀錄"), C.AMBER)
        en = ({"utt": log["english"]["utt"], "text": log["english"]["text"], "names": log["english"]["names"]}, None, None) if log.get("english") else None
        zh = ({"text": plog["mandarin"]["text"], "names": plog["mandarin"]["names"]}, None, None) if plog.get("mandarin") else None
        d = Demo(out, en, zh, FakeWhisper(), FakeCTC(), FakeQwen(), stored=log["results"], stored_private=plog.get("results", {}), tag="replay")
        d.prompts_en = log.get("prompts_en", d.prompts_en)
        if plog.get("prompts_zh"):
            d.prompts_zh = plog["prompts_zh"]
        return d
    en = load_english()
    zh = load_mandarin()
    if zh is None:
        C.note("  （還沒有自錄的華語句：./present.sh record；3–6、8、9 的華語那半會跳過）")
    if cmd == "fake":
        out.banner("FAKE：罐頭輸出，只為了走流程，不是模型的輸出", C.RED)
        return Demo(out, en, zh, FakeWhisper(), FakeCTC(), FakeQwen(), tag="fake")
    t0 = time.perf_counter()
    wh = Whisper()
    ctc = CTC()
    qw, err = None, None
    try:
        qw = Qwen()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if pick_device(CFG["qwen"]["device"]).type == "mps":
            try:
                C.note(f"  Qwen3-ASR 在 MPS 上載不起來（{err[:80]}），退 CPU")
                qw, err = Qwen("cpu"), None
            except Exception as e2:
                err = f"{type(e2).__name__}: {e2}"
    C.note(f"  Whisper small {wh.n_params / 1e6:.0f} M（{wh.device}）、wav2vec2-base-960h {ctc.n_params / 1e6:.0f} M（{ctc.device}）、"
           + (f"Qwen3-ASR {qw.n_params / 1e9:.1f} B（{qw.device}，{qw.dtype}）" if qw else f"Qwen3-ASR 載入失敗：{err}") + f"，載入 {time.perf_counter() - t0:.1f} s")
    d = Demo(out, en, zh, wh, ctc, qw, tag="rehearsal" if cmd == "rehearse" else "")
    d.qw_error = err
    return d


# ── check ───────────────────────────────────────────────────────────────────
def check() -> None:
    C.banner("課前體檢", C.AMBER)
    res: dict = {"time": C.now()}
    d = Demo(Out(), None, None, FakeWhisper(), FakeCTC(), FakeQwen())
    v = d.versions()
    for k in ("python", "platform", "numpy", "scipy", "torch", "transformers", "huggingface_hub", "models"):
        print(f"  {k:<16s} {v.get(k)}")
    res["versions"] = v
    try:
        import torch
    except ImportError:
        print("  沒有 torch：先在 w03_d3_blank_heatmap 跑 ./setup.sh"); return
    mps = bool(torch.backends.mps.is_available())
    res["mps_available"] = mps
    print(f"  {'mps':<16s} available={mps}")
    for repo in (CFG["whisper"]["repo"], CFG["ctc"]["repo"], CFG["qwen"]["repo"]):
        try:
            print(f"  {'model':<16s} {repo} → {model_dir(repo)}")
        except Exception as e:
            print(f"  {'model':<16s} {repo} 不在快取：{type(e).__name__}\n  先跑 ./present.sh fetch"); return
    for mod in ("pyctcdecode", "kenlm", "soundfile", "librosa", "torchcodec"):
        try:
            from importlib import metadata
            __import__(mod)
            res.setdefault("packages", {})[mod] = metadata.version(mod)
        except Exception as e:
            res.setdefault("packages", {})[mod] = f"無（{type(e).__name__}）"
    print(f"  {'packages':<16s} {res['packages']}（pyctcdecode 沒裝也沒關係：hotword 用自寫的 ctc_prefix_beam.py）")
    ref, wav, x = load_english()
    zh = load_mandarin()
    res["english"] = {"utt": ref["utt"], "names": ref["names"], "seconds": len(x) / FS}
    res["mandarin_recorded"] = zh is not None
    print(f"  英文那句 {ref['utt']}（{len(x) / FS:.2f} s）；華語自錄句：{'有' if zh else '還沒錄（./present.sh record）'}")

    # 1. Whisper：prompt 槽通不通、名單對命中有沒有影響
    wc = Whisper()
    r0 = wc.transcribe(x, "en", None)
    r1 = wc.transcribe(x, "en", CFG["prompts"]["list_fmt_en"].format(names=fmt_list(ref["names"], False)))
    for tag, r in (("無 prompt", r0), ("名單", r1)):
        h = name_hits(r["text"], ref["names"], False)
        print(f"  whisper {tag:<8s} {r['text']!r}  名字 {sum(h.values())}/{len(h)}  avg {r['avg_logprob']:.3f}  {r['seconds']:.2f} s")
    res["whisper"] = {"none": r0, "list": r1}

    # 2. CTC：prefix beam δ=0 對 greedy；trie 的 bonus 真的加進去
    cc = CTC()
    lp = cc.logprobs(x)
    g = cc.decode(ctc_greedy(lp, cc.blank))
    b0 = prefix_beam_search(lp, cc.blank, beam=int(CFG["ctc"]["beam"]), prune_logp=float(CFG["ctc"]["prune_logp"]))[0]
    trie = Trie([cc.encode(n) for n in ref["names"]])
    dmax = float(CFG["ctc"]["deltas"][-1])
    bD = prefix_beam_search(lp, cc.blank, beam=int(CFG["ctc"]["beam"]), bonus=HotwordBonus(trie, dmax, cc.delim), prune_logp=float(CFG["ctc"]["prune_logp"]))[0]
    print(f"  ctc greedy      {g!r}\n  ctc beam δ=0    {cc.decode(b0['ids'])!r}  log P {b0['logp']:.3f}\n  ctc beam δ={dmax:g}  {cc.decode(bD['ids'])!r}  log P {bD['logp']:.3f} + bonus {bD['bonus']:.1f}")
    print(f"  符號表：blank {cc.blank}、詞界 {cc.delim}（{cc.proc.tokenizer.word_delimiter_token!r}）；名單編碼 {[cc.encode(n) for n in ref['names']]}")
    res["ctc"] = {"device": str(cc.device), "T": int(lp.shape[0]), "greedy": g, "beam0": cc.decode(b0["ids"]), "beam0_logp": b0["logp"],
                  "beam_dmax": cc.decode(bD["ids"]), "beam_dmax_logp": bD["logp"], "beam_dmax_bonus": bD["bonus"], "blank": cc.blank, "delim": cc.delim}

    # 3. Qwen3-ASR：MPS 載得起來嗎、generate 通不通、prompt 參數吃不吃、model card 的 Quilter 例句
    res["qwen"] = {}
    w3 = _p(CFG["audio"]["w3_wav"])
    x3 = C.load_wav(w3, FS) if w3.exists() else None
    for dev in (["mps", "cpu"] if mps else ["cpu"]):
        try:
            t0 = time.perf_counter(); q = Qwen(dev); tl = time.perf_counter() - t0
            r = q.transcribe(wav, x, "English", None)
            rl = q.transcribe(wav, x, "English", CFG["prompts"]["qwen_list_fmt"].format(names=fmt_list(ref["names"], False)))
            h0, h1 = name_hits(r["text"], ref["names"], False), name_hits(rl["text"], ref["names"], False)
            entry = {"load_s": tl, "dtype": str(q.dtype), "none": r, "list": rl, "hits_none": sum(h0.values()), "hits_list": sum(h1.values())}
            print(f"  qwen3-asr {dev:<4s} 載入 {tl:.1f} s（{q.dtype}）；英文無 context {r['text']!r}（名字 {sum(h0.values())}/{len(h0)}，{r['seconds']:.2f} s，音訊給法 {r['audio_as']}）")
            print(f"  qwen3-asr {dev:<4s} 名單 context    {rl['text']!r}（名字 {sum(h1.values())}/{len(h1)}）")
            if x3 is not None:
                rq = q.transcribe(w3, x3, "English", "Vocabulary: Quilter, apostle, gospel.")
                entry["w3_modelcard_prompt"] = rq
                print(f"  qwen3-asr {dev:<4s} W3 那句 + model card 的 context：{rq['text']!r}")
            res["qwen"][dev] = entry
            del q
            break                                           # 第一條能跑的就是課堂用的
        except Exception as e:
            res["qwen"][dev] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  qwen3-asr {dev:<4s} 失敗：{type(e).__name__}: {str(e)[:160]}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    C.note("  結果存在 runs/check.json（進版控；只有英文那句的文字）")


# ── pick ────────────────────────────────────────────────────────────────────
def pick() -> None:
    C.banner("挑英文那句：每個候選過 Whisper 無 prompt 與 CTC greedy，看名字錯不錯", C.AMBER)
    cands = CFG["english"]["candidates"]
    missing = [c["utt"] for c in cands if not english_wav_path(c["utt"]).exists()]
    if missing:
        print(f"  解 wav：{missing}")
        extract_from_tar(missing)
    wc, cc = Whisper(), CTC()
    rows = []
    for c in cands:
        p = english_wav_path(c["utt"])
        if not p.exists():
            print(f"  {c['utt']} 沒有 wav，跳過"); continue
        x = C.load_wav(p, FS)
        rw = wc.transcribe(x, "en", None)
        g = cc.decode(ctc_greedy(cc.logprobs(x), cc.blank))
        hw, hc = name_hits(rw["text"], c["names"], False), name_hits(g, c["names"], False)
        mark = "★" if (sum(hw.values()) < len(hw) or sum(hc.values()) < len(hc)) else " "
        print(f"\n {mark} {c['utt']}  {len(x) / FS:.1f} s  名單 {c['names']}")
        print(f"    whisper {rw['text']!r}  名字 {sum(hw.values())}/{len(hw)}")
        print(f"    ctc     {g!r}  名字 {sum(hc.values())}/{len(hc)}")
        rows.append({"utt": c["utt"], "names": c["names"], "whisper": rw["text"], "whisper_hits": sum(hw.values()), "ctc": g, "ctc_hits": sum(hc.values())})
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "pick.json").write_text(json.dumps({"time": C.now(), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  ★ = 至少一個模型漏了一個名字。選好就把 toml [english] utt 改成它（目前 {CFG['english']['utt']}）。結果在 runs/pick.json（進版控）")


# ── fetch／record／export ─────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載模型與解英文候選句", C.AMBER)
    lock = read_lock()
    lines = []
    for repo in (CFG["whisper"]["repo"], CFG["ctc"]["repo"], CFG["qwen"]["repo"]):
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        path = snapshot_download(repo, revision=sha, allow_patterns=PATTERNS)
        lines.append(f"{repo} {sha}")
        print(f"  {repo} @ {sha[:12]} → {path}")
    MODEL_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  commit 寫進 model.lock（Whisper、wav2vec2 與 Demo 1／W4 d3／W3 d3 同一個；Qwen3-ASR 是這次抓到的）")
    missing = [c["utt"] for c in CFG["english"]["candidates"] if not english_wav_path(c["utt"]).exists()]
    if missing:
        done = extract_from_tar(missing)
        print(f"  從 dev-clean.tar.gz 解出 {done}" + (f"；解不到 {sorted(set(missing) - set(done))}" if set(missing) - set(done) else ""))
    else:
        print("  英文候選句的 wav 都在（W5 Demo 1 解過）")
    print("\n下一步：./present.sh pick → 改 toml 的 utt → ./present.sh record → ./present.sh check → ./present.sh rehearse")


def record() -> None:
    m = CFG["mandarin"]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    C.banner(f"錄 {m['seconds']:g} 秒華語，一句話、裡面放三四個專有名詞（人名、路名、產品名）", C.AMBER)
    x = C.record(float(m["seconds"]), FS)
    C.save_wav(PRIVATE / "mandarin.wav", x, FS)
    print(f"  → {PRIVATE}/mandarin.wav（不進版控、不上站）。想聽：afplay {PRIVATE}/mandarin.wav")
    ref = input("  參考文字（繁體，照講的打；Enter 跳過）：").strip()
    if ref:
        (PRIVATE / "mandarin.txt").write_text(ref + "\n", encoding="utf-8")
    names = input("  名單（繁體，用「、」或空白分開；Enter 跳過）：").strip()
    if names:
        nl = [n for n in re.split(r"[、,\s]+", names) if n]
        (PRIVATE / "names.txt").write_text("\n".join(nl) + "\n", encoding="utf-8")
        print(f"  名單 {nl} → names.txt")
    print("  之後要改就直接編輯 mandarin.txt、names.txt。")


def export() -> None:
    src = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    if log.get("backend") != "whisper":
        sys.exit(f"rehearsal.json 的 backend 是 {log.get('backend')}，不是真模型的紀錄，不匯出")
    psrc = PRIVATE / "private_rehearsal.json"
    plog = json.loads(psrc.read_text(encoding="utf-8")) if psrc.exists() else {"results": {}}
    R, RP = log["results"], plog["results"]

    def pub(r):
        if not r:
            return None
        return {"text": r["text"], "n_hit": r["n_hit"], "n_names": r["n_names"], "hits": r["hits"], "leak": r["leak"], "avg_logprob": r.get("avg_logprob"),
                "wer": r.get("wer"), "seconds": r.get("seconds")}

    def priv(r):                                    # 華語：只進數字，不進文字
        if not r:
            return None
        return {"n_hit": r["n_hit"], "n_names": r["n_names"], "script": r["script"]["verdict"] if r.get("script") else None,
                "leaked": bool(r["leak"]), "n_leak_fragments": len(r["leak"]), "n_chars": len(r["text"]), "seconds": r.get("seconds")}
    ctc = R.get("ctc:en")
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    data["demo2"] = {
        "source": "demos/w07_d2_biasing/runs/rehearsal/rehearsal.json (+ the instructor's private Mandarin record, counts only)",
        "time": log["time"], "device": log["device"], "models": log["versions"].get("models"), "transformers": log["versions"].get("transformers"), "torch": log["versions"].get("torch"),
        "english": log.get("english"), "prompts_en": log.get("prompts_en"),
        "whisper_en": {k: pub(R.get(f"whisper:en:{k}")) for k in ("none", "list", "dialogue")},
        "qwen_en": {k: pub(R.get(f"qwen:en:{k}")) for k in ("none", "list", "dialogue")},
        "qwen_error": log.get("qwen_error"),
        "ctc_en": {"T": ctc["T"], "deltas": CFG["ctc"]["deltas"], "beam": CFG["ctc"]["beam"],
                   "rows": [{"label": r["label"], "delta": r["delta"], "text": r["text"], "n_hit": r["n_hit"], "wer": r["wer"], "logp": r["logp"], "bonus": r["bonus"],
                             "n_changed_vs_delta0": r["diff_vs_delta0"]["n_changed"] if r.get("diff_vs_delta0") else None,
                             "changes_vs_delta0": r["diff_vs_delta0"]["changes"] if r.get("diff_vs_delta0") else None} for r in ctc["rows"]]} if ctc else None,
        "quiet": {k[6:]: {"text": v["text"], "leak": v["leak"], "echo_verbatim": v.get("echo_verbatim"), "avg_logprob": v.get("avg_logprob"), "seconds": v.get("seconds")}
                  for k, v in R.items() if k.startswith("quiet:")},
        "mandarin": {"n_names": log.get("mandarin", {}).get("n_names"),
                     "whisper": {k: priv(RP.get(f"whisper:zh:{k}")) for k in ("none", "list", "traditional", "dialogue")},
                     "qwen": {k: priv(RP.get(f"qwen:zh:{k}")) for k in ("none", "list", "traditional", "dialogue")}},
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    w = data["demo2"]["whisper_en"]
    print(f"→ {METRICS}（demo2）：英文 Whisper 無 prompt {w['none']['n_hit'] if w['none'] else '—'}/{len(log['english']['names'])} → 名單 {w['list']['n_hit'] if w['list'] else '—'}；"
          f"台詞洩漏 {w['dialogue']['leak'] if w['dialogue'] else '—'}")


# ── selftest ───────────────────────────────────────────────────────────────
def selftest() -> None:
    global PRIVATE
    C.banner("selftest（不需要模型）", C.AMBER)
    fails = []

    def chk(name: str, cond: bool, extra: str = "") -> None:
        print(f"  {'ok ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)
    fc = FakeCTC()
    # trie 與 failure arc
    trie = Trie([fc.encode("Glenarvan"), fc.encode("Mary Grant")])
    hb = HotwordBonus(trie, 1.0, fc.delim)
    s = hb.start()
    for c in fc.encode("GLENARVAN"):
        s = hb.step(s, c)
    chk("走完 GLENARVAN：partial 9", abs(s[1] - 9.0) < 1e-12 and s[0] in trie.terminal)
    s2 = hb.step(s, fc.delim)
    chk("接到詞界：committed 9、回 root", abs(s2[2] - 9.0) < 1e-12 and s2[0] == 0)
    chk("句尾沒接詞界也算走完", abs(hb.finish(s) - 9.0) < 1e-12)
    s = hb.start()
    for c in fc.encode("GLEN"):
        s = hb.step(s, c)
    s = hb.step(s, fc.vocab["X"])
    chk("GLENX 岔開：failure arc 扣回、進 OFF", s[1] == 0.0 and s[0] == -1 and s[2] == 0.0)
    s = hb.step(s, fc.delim)
    chk("詞界後回 root", s[0] == 0)
    s = hb.start()
    for c in fc.encode("XGLEN"):
        s = hb.step(s, c)
    chk("不從詞界開始不比對（XGLEN 沒有加分）", hb.score(s) == 0.0)
    s = hb.start()
    for c in fc.encode("MARY GRANT"):
        s = hb.step(s, c)
    chk("多詞片語 MARY GRANT：含中間的詞界共 10 格", abs(s[1] - 10.0) < 1e-12 and s[0] in trie.terminal)
    s = hb.start()
    for c in fc.encode("MARY SMITH"):
        s = hb.step(s, c)
    chk("MARY SMITH：MARY 不是獨立片語，整段扣回", hb.finish(s) == 0.0)
    # prefix beam search 對窮舉（小例子：T=2，blank .4／a .35／b .25 兩格：greedy 給空、beam 給 a）
    lp = np.log(np.array([[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]]))
    best = prefix_beam_search(lp, 0, beam=8, prune_logp=-50)
    chk("prefix beam：T=2 的例子最好的是 'a'", best[0]["ids"] == [1], f"log P {best[0]['logp']:.4f}")
    chk("prefix beam：P('a') = .35·.4 + .4·.35 + .35·.35 = .4025", abs(np.exp(best[0]["logp"]) - 0.4025) < 1e-9)
    chk("greedy：同一例子給空字串", ctc_greedy(lp, 0) == [])
    pe = [b for b in best if b["ids"] == []]
    chk("prefix beam：P('') = .16", pe and abs(np.exp(pe[0]["logp"]) - 0.16) < 1e-9)
    # 重複字元：'aa' 要靠中間的 blank
    lp2 = np.log(np.array([[0.1, 0.9, 0.0 + 1e-9], [0.9, 0.1, 1e-9], [0.1, 0.9, 1e-9]]))
    best2 = prefix_beam_search(lp2, 0, beam=8, prune_logp=-50)
    chk("prefix beam：a blank a → 'aa' 最好", best2[0]["ids"] == [1, 1], f"{best2[0]['ids']}")
    # 合成 logits：δ=0 給 GLENARVIN、δ 大給 GLENARVAN
    lpf = fc.logprobs(None)
    b0 = prefix_beam_search(lpf, fc.blank, beam=16)
    chk("合成例子 δ=0：GLENARVIN", "GLENARVIN" in fc.decode(b0[0]["ids"]), fc.decode(b0[0]["ids"]))
    bD = prefix_beam_search(lpf, fc.blank, beam=16, bonus=HotwordBonus(trie, 3.0, fc.delim))
    chk("合成例子 δ=3：改成 GLENARVAN", "GLENARVAN" in fc.decode(bD[0]["ids"]), fc.decode(bD[0]["ids"]))
    chk("δ=3 的 bonus = 9 + 10", abs(bD[0]["bonus"] - 57.0) < 1e-9, f"{bD[0]['bonus']}")
    # 判定函數
    chk("名字命中（英文、詞界）", name_hits("Think of Lady Glenarvan, think of Mary Grant.", ["Glenarvan", "Mary Grant"], False) == {"Glenarvan": True, "Mary Grant": True})
    chk("名字不命中（Glenarvin）", name_hits("Lady Glenarvin", ["Glenarvan"], False)["Glenarvan"] is False)
    chk("名字命中（華語、繁簡不計）", name_hits("我们去台北找王老师", ["臺北", "王老師"], True) == {"臺北": True, "王老師": True})
    chk("字體：簡", script_judge("我们今天开会")["verdict"] == "簡")
    chk("字體：繁", script_judge("我們今天開會")["verdict"] == "繁")
    chk("字體：無法判定", script_judge("你好")["verdict"] == "無法判定")
    chk("洩漏（英文 3 詞，合成最長片段）", leak("Thanks for watching, and don't forget to subscribe.", "Don't forget to subscribe to this.", False, 3) == ["don't forget to subscribe"])
    chk("沒洩漏", leak("Thanks for watching.", "Think of Mary.", False, 3) == [])
    chk("洩漏（華語 4 字，繁簡不計）", leak("謝謝大家，下次見。", "好，谢谢大家下次见", True, 4) == ["谢谢大家下次见"])
    chk("詞級 diff", word_diff("think of lady glenarvin", "think of lady glenarvan")["changes"] == [{"from": "glenarvin", "to": "glenarvan"}])
    chk("WER", wer_words("A B C", "A X C D") == {"errors": 2, "n_ref": 3, "wer": 2 / 3})
    dz = Demo(Out(), None, ({"text": "", "names": ["臺北市長"]}, Path("x"), np.ones(FS)), FakeWhisper(), FakeCTC(), FakeQwen())
    chk("名單 prompt 命中不算洩漏", dz._annotate({"text": "這次臺北市長大選"}, {"names": ["臺北市長"]}, True, "蔣萬安、臺北市長。")["leak"] == [])
    # fake 整條流程
    out = Out()
    speech = np.random.default_rng(0).standard_normal(FS) * 0.1                # fake 後端只看「不是近乎靜音」
    en = ({"utt": "2803-154320-0011", "text": "THINK OF LADY GLENARVAN THINK OF MARY GRANT", "names": ["Glenarvan", "Mary Grant"]}, Path("fake.wav"), speech)
    zh = ({"text": "我們今天要去臺北的實驗室找王老師。", "names": ["臺北", "王老師"]}, Path("fake.wav"), speech)
    d = Demo(out, en, zh, FakeWhisper(), FakeCTC(), FakeQwen(), tag="selftest")
    try:
        d.run_all()
        for k in "12e34567890pth":
            d.dispatch(k)
        chk("fake 整條流程跑完", True, f"{len(d.res)} 公開、{len(d.res_priv)} 私密")
        chk("華語結果不在公開水槽", not any(k.split(':')[1] == 'zh' for k in d.res) and all(k.split(':')[1] == 'zh' for k in d.res_priv))
        chk("公開 txt 裡沒有華語參考句", "王老師" not in out.text() and "王老師" in out.text(private=True))
        chk("名單 prompt 修好 Glenarvan（fake）", d.res["whisper:en:list"]["n_hit"] == 2 and d.res["whisper:en:none"]["n_hit"] == 1)
        chk("台詞 prompt 洩漏（fake）", bool(d.res["whisper:en:dialogue"]["leak"]))
        chk("繁體 prompt 翻字體（fake）", d.res_priv["whisper:zh:traditional"]["script"]["verdict"] == "繁" and d.res_priv["whisper:zh:none"]["script"]["verdict"] == "簡")
        chk("CTC 格：greedy + δ=0 + 三個 δ", len(d.res["ctc:en"]["rows"]) == 5)
        chk("近乎靜音那格：六個結果", sum(1 for k in d.res if k.startswith("quiet:")) == 6)
        chk("quiet_clip 確定性、RMS 對", np.array_equal(quiet_clip(), quiet_clip()) and abs(float(np.sqrt(np.mean(quiet_clip() ** 2))) - CFG["audio"]["quiet_rms"]) < 1e-9)
        rp = Demo(Out(), en, zh, FakeWhisper(), FakeCTC(), FakeQwen(), stored=d.res, stored_private=d.res_priv)
        chk("replay 從兩份 results 重建", rp.dispatch("1") and rp.dispatch("3") and rp.dispatch("7"))
        tmp = HERE / "runs" / "_selftest"
        old_priv = PRIVATE
        PRIVATE = tmp / "_private"
        d.save(tmp)
        j = json.loads((tmp / "rehearsal.json").read_text(encoding="utf-8"))
        chk("rehearsal.json 可讀、沒有華語文字", j["backend"] == "fake" and "王老師" not in (tmp / "rehearsal.json").read_text(encoding="utf-8"))
        chk("private_selftest.json 有華語", "王老師" in (PRIVATE / "private__selftest.json").read_text(encoding="utf-8"))
        PRIVATE = old_priv
        import shutil
        shutil.rmtree(tmp)
    except Exception as e:
        import traceback
        traceback.print_exc()
        chk("fake 整條流程", False, f"{type(e).__name__}: {e}")
    print(f"\n  {'全部通過' if not fails else '失敗：' + ', '.join(fails)}")
    if fails:
        sys.exit(1)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "pick", "fetch", "record", "fake", "selftest", "export"])
    args = ap.parse_args()
    if args.cmd == "check": check(); return
    if args.cmd == "pick": pick(); return
    if args.cmd == "fetch": fetch(); return
    if args.cmd == "record": record(); return
    if args.cmd == "selftest": selftest(); return
    if args.cmd == "export": export(); return

    out = Out()
    d = build(args.cmd, out)
    if args.cmd == "rehearse":
        d.run_all()
        d.save(HERE / "runs" / "rehearsal")
        print("\n下一步：./present.sh 自己按一遍 1 2 e 3 4 5 6 7 8 9 0 t；把 rehearsal.json（英文）與 ../_media/w07_d2/private_rehearsal.json（華語）的結果填進 p50 的表、"
              ".gen 區塊、【操作】【講述】；./present.sh export 寫進 w07-metrics.json。")
        return
    out.banner("W7 Demo 2：Biasing——同一段音訊，有名單與沒有名單" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    while True:
        k = C.getkey()
        if not d.dispatch(k):
            break
    if args.cmd == "show":
        d.save(HERE / "runs" / "live" / C.now().replace(":", "-"))


if __name__ == "__main__":
    main()
