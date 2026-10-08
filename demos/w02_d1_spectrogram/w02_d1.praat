# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W2 demo 1：同一段音的 wideband、narrowband spectrogram 與 pitch track，畫在 Picture 視窗。
#
# 用法（兩種）：
#   1. Praat 選單 Praat > Open Praat script… 開這個檔，Run，在表單裡填 WAV 的完整路徑
#   2. 終端機：./present.sh praat vowels
#
# **這支 script 是 2026-09-17 憑對 Praat 語法的記憶寫的，沒有在 Praat 裡跑過。**
# 第一次用之前先跑一次；指令名稱或參數順序若與你的 Praat 版本不合，Praat 會指出是哪一行。
# 最容易出錯的是 Paint 與 To Pitch 的參數個數——對照 Praat 裡按同名按鈕跳出來的表單欄位即可。

form W2 demo 1
    sentence Wav_file runs/rehearsal/vowels.wav
    positive Pitch_floor_(Hz) 75
    positive Pitch_ceiling_(Hz) 400
endform

sound = Read from file: wav_file$
Erase all
Font size: 12

# ── 上：wideband，窗長 5 ms（Praat 的預設）。看得到聲門脈衝的直條紋與共振峰的橫帶
Select outer viewport: 0, 9, 0, 3
selectObject: sound
wide = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"
Paint: 0, 0, 0, 0, 100, "yes", 50, 6, 0, "yes"
Text top: "no", "WIDEBAND  window = 5 ms"

# ── 中：narrowband，窗長 30 ms。直條紋消失，換成一條條水平的諧波
Select outer viewport: 0, 9, 3, 6
selectObject: sound
narrow = To Spectrogram: 0.03, 5000, 0.002, 20, "Gaussian"
Paint: 0, 0, 0, 0, 100, "yes", 50, 6, 0, "yes"
Text top: "no", "NARROWBAND  window = 30 ms"

# ── 下：pitch track。floor／ceiling 設錯會出現 octave error——故意示範時把 ceiling 改成 150
Select outer viewport: 0, 9, 6, 9
selectObject: sound
pitch = To Pitch: 0, pitch_floor, pitch_ceiling
Draw: 0, 0, 0, pitch_ceiling, "yes"
Text top: "no", "F0 (Hz)"

# 留下 Sound 物件並開編輯視窗，要讀共振峰數值時用（Formant > Show formants，游標放上去按 F1）
removeObject: wide, narrow, pitch
selectObject: sound
View & Edit
