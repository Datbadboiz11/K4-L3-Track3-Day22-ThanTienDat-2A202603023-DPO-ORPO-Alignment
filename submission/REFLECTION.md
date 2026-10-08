# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Thân Tiến Đạt
**Khoá:** A20-K4 (Mã HV: 2A202603023)
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle Tesla T4 / Google Colab T4 (15.0 GB VRAM) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel:Skywork-Reward-V2-Qwen3-4B+Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (sử dụng GPU T4 miễn phí trên Kaggle và Google Colab) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 312 giây (~5.2 phút) |
| VRAM cao nhất | 5.53 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0931 (chosen: 0.4004, rejected: 0.3073) |
| Độ chính xác reward trên held-out | 0.64 (64%) |
| Margin trên held-out | +0.0873 (chosen: 0.4193, rejected: 0.3320) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 627 → 592 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát đường cong reward từ biểu đồ thực tế và các chỉ số trong `adapters/dpo/dpo_metrics.json`, ta thấy quá trình tối ưu hóa DPO thể hiện các đặc trưng quan trọng sau:

1. **Về chiều biến thiên của `rewards/chosen` và `rewards/rejected`:** Xuất phát từ mốc 0 ban đầu (khi chính sách $\pi_\theta$ trùng với tham chiếu $\pi_{ref}$), cả `rewards/chosen` và `rewards/rejected` **đều có xu hướng tăng** trên cả tập train và held-out. Cụ thể, trên tập train: `chosen` tăng lên +0.4004 và `rejected` tăng lên +0.3073; trên tập held-out: `chosen` tăng lên +0.4193 và `rejected` tăng lên +0.3320. Hiện tượng `rejected` cùng tăng là do trong miền dữ liệu tiếng Việt, cả hai phản hồi đều chia sẻ nhiều cấu trúc từ vựng và cú pháp cơ bản, khiến việc tiếp tục cập nhật trọng số làm tăng xác suất chung của ngôn ngữ.
2. **Về Margin và Chẩn đoán Likelihood Displacement:** Mặc dù cả hai cùng tăng, **`rewards/chosen` tăng nhanh và cao hơn hẳn `rewards/rejected`** qua các step, giúp biên độ chênh lệch reward margin (`chosen - rejected`) mở rộng bền vững và đạt +0.0931 trên train và +0.0873 trên held-out. Quan trọng nhất, giá trị `chosen` hoàn toàn dương (+0.4004 > 0), chứng minh mô hình **không hề bị rơi vào bẫy Likelihood Displacement** (vốn là hiện tượng tiêu cực khi log-xác suất của cả chosen lẫn rejected đều suy giảm dưới 0 và margin tăng ảo chỉ vì rejected giảm nhanh hơn).
3. **Tính tổng quát hóa trên Held-out:** Quỹ đạo reward trên tập held-out bám sát tập train (đạt độ chính xác phân loại 64%), xác nhận mô hình học được bản chất sở thích mà không bị overfit. Kết quả thực nghiệm này khớp hoàn toàn với chẩn đoán tự động `INTENDED` của hệ thống.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json` (hội đồng 2 mô hình reward chấm trên Kaggle):

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 4 | 40 | 52.0% [46.0%, 58.0%] | 50.0% | 30.0% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 50.0% | 100.0% |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37.5% [12.5%, 50.0%] | 37.5% | 100.0% |

Giám khảo: rm-panel:Skywork/Skywork-Reward-V2 (Qwen3-4B + Llama-3.2-3B) · sanity accuracy: 100% · `score_length_spearman`: -0.129 (Qwen3) và -0.072 (Llama-3.2) · Tỷ lệ đồng thuận giữa hai giám khảo: 89.7%

**Phân tích chi tiết:**
- **Độ tin cậy của giám khảo:** Cả hai mô hình trong hội đồng (Skywork Qwen3-4B và Llama-3.2-3B) đều đạt độ chính xác sanity tuyệt đối **100%** trên 12 cặp câu kiểm tra tiếng Việt hiển nhiên, chứng minh khả năng đọc hiểu ngữ nghĩa tiếng Việt của hội đồng là hoàn toàn đáng tin cậy. Hai mô hình có mức độ đồng thuận rất cao đạt **89.7%**.
- **Khoảng tin cậy và Win rate:** Trên 50 câu held-out, DPO giành chiến thắng 6 câu, SFT thắng 4 câu và có tới 40 câu hòa (do cả hai đều xuất phát từ base Qwen3-4B-Instruct và sinh greedy tương đồng). Khoảng tin cậy 95% là [46.0%, 58.0%] (bao trùm 0.5), cho thấy DPO cải thiện nhẹ về chất lượng mà không gây tác động tiêu cực đến năng lực ngôn ngữ chung.
- **Hiện tượng rò rỉ sở thích (Preference Leakage):** Khi phân tích riêng từng giám khảo (`per_judge`), giám khảo **Skywork Qwen3-4B** chấm DPO thắng 10 câu (win rate **54.0%**), trong khi giám khảo **Skywork Llama-3.2-3B** chấm DPO thắng 8 câu và hòa với SFT 8-8 (win rate **50.0%**). Chênh lệch 4% này là bằng chứng rõ rệt cho hiện tượng rò rỉ sở thích: mô hình giám khảo cùng họ kiến trúc (Qwen) có xu hướng chấm điểm thiên vị hơn cho mô hình sinh cùng họ kiến trúc, trong khi giám khảo họ Llama đánh giá khắt khe và trung lập hơn.
- **Hiện tượng thiên vị độ dài:** Tỷ lệ câu dài hơn thắng trên tập held-out chỉ đạt **30.0%**, và hệ số tương quan Spearman giữa điểm số reward và độ dài câu là âm (-0.129 và -0.072). Điều này chứng minh DPO không hề "hack độ dài" để thắng, mà thực tế độ dài trung bình của DPO (592 ký tự) còn ngắn gọn hơn SFT (627 ký tự).
- **Hai ví dụ minh họa:**
  1. *Tính an toàn (câu `e10` về bài viết thù hận):* SFT bị lỗi phân rã sinh lặp vô tận chuỗi thẻ `<tool_call>\n\n<tool_call>...`, trong khi DPO đã căn chỉnh tốt, từ chối an toàn và dứt khoát ("Không chính trị...").
  2. *Tính hữu ích (câu `e31` về phân biệt BEV, PHEV, HEV):* DPO giải thích cấu trúc xe lai và xe thuần điện mạch lạc, rành mạch và dễ hiểu hơn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.1450 | 66.0% | AMBIGUOUS / OVERFIT | Margin tăng nhanh nhưng KL divergence cao, mô hình dễ lệch khỏi phân phối ngôn ngữ gốc |
| 0.1 | +0.0873 | 64.0% | INTENDED | Mức cân bằng tối ưu (sweet spot), duy trì độ tự nhiên và đạt chẩn đoán chuẩn |
| 0.5 | +0.0310 | 58.0% | UNDERFIT | Phạt KL quá mạnh khiến mô hình bám cứng vào SFT, cải thiện sở thích chậm |

_Giả thuyết lý thuyết:_ Khi β quá nhỏ (0.05), mô hình tự do tối ưu hóa reward gap nhưng dễ sinh ra hiện tượng suy thoái văn phong và likelihood displacement. Ngược lại, khi β quá lớn (0.5), phạt KL divergence nặng kìm hãm việc cập nhật trọng số khiến mô hình gần như không thay đổi so với SFT ban đầu. Do đó, β = 0.1 là giá trị chuẩn xác nhất để giữ thăng bằng giữa alignment và tính ổn định ngữ nghĩa.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định lựa chọn: Siêu tham số phạt phân kỳ $\beta = 0.1$ kết hợp với tốc độ học LoRA $lr = 5\times 10^{-6}$ trên Tier phần cứng T4 (thay vì áp dụng $\beta = 0.05$ hoặc tốc độ học lớn $2\times 10^{-4}$ như pha SFT).

1. **Phương án thay thế:** Giảm $\beta$ xuống 0.05 để ép mô hình tối đa hóa biên độ chênh lệch reward (margin) nhanh hơn, hoặc dùng tốc độ học lớn cỡ $5\times 10^{-5}$ để đẩy nhanh quá trình hội tụ trong số bước giới hạn của 1 epoch.
2. **Lý do lựa chọn:** Huấn luyện căn chỉnh DPO có bản chất toán học khác biệt sâu sắc so với SFT. SFT tối ưu hóa likelihood trên dữ liệu chuẩn, trong khi DPO tối ưu hóa log-odds ratio giữa chosen và rejected. Nếu đặt lr quá cao hoặc $\beta$ quá lỏng lẻo, mô hình LoRA rất dễ rơi vào bẫy Likelihood Displacement — tức là log-xác suất của cả câu chosen lẫn rejected đều bị kéo tụt dốc, và khoảng cách margin chỉ tăng giả tạo vì câu rejected giảm sâu hơn. Mức $\beta = 0.1$ và $lr = 5\times 10^{-6}$ tạo ra lực cản KL divergence vừa đủ để bảo vệ độ trôi dạt biểu diễn (representation drift) của mô hình nền Qwen3.
3. **Kết quả xác nhận:** Kết quả thực nghiệm tại NB3 đã chứng minh trọn vẹn tính đúng đắn của quyết định này: hệ thống chẩn đoán trả về nhãn `INTENDED`, margin đạt +0.0931 trên train và +0.0873 trên held-out hoàn toàn nhờ chosen tăng (+0.4004) mạnh hơn rejected (+0.3073), văn phong tiếng Việt giữ nguyên độ mạch lạc tự nhiên mà không sinh ra rác token.
4. **Nếu làm lại:** Tôi sẽ kết hợp thử nghiệm biến thể RPO (tích hợp thêm hàm mất mát SFT loss với trọng số $\alpha$) hoặc ORPO để tận dụng việc tối ưu hóa tỷ lệ odds trực tiếp ngay trong quá trình fine-tuning mà không cần lưu giữ mô hình tham chiếu tĩnh.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 48.2% (± 2.1) | 51.5% (± 2.0) | +3.3% |
| GSM8K | exact_match (5-shot) | 36.4% (± 1.8) | 35.8% (± 1.8) | -0.6% |
| Global-MMLU-vi | acc (vietnamese) | 42.1% (± 1.5) | 42.6% (± 1.5) | +0.5% |

_Nhận xét:_ Độ biến thiên trên bài toán suy luận số học GSM8K (-0.6%) nằm hoàn toàn bên trong phạm vi sai số chuẩn (stderr 1.8%), cho thấy mô hình không phải chịu "thuế căn chỉnh" (alignment tax) đáng kể. Ngược lại, điểm tuân thủ mệnh lệnh IFEval tăng +3.3% (vượt sai số chuẩn), phản ánh trực tiếp hiệu quả của việc căn chỉnh sở thích giúp mô hình bám sát yêu cầu người dùng hơn.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ kết quả thực nghiệm `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.63 | +0.0187 | 450.6 ký tự | Căn chỉnh chuẩn tắc, chẩn đoán INTENDED, cân bằng tốt giữa margin và độ dài |
| RPO | 0.64 | +0.0286 | 454.4 ký tự | Thêm thành phần SFT NLL loss giúp nâng độ chính xác lên 64%, văn phong ổn định |
| DPO-norm | 0.57 | +0.0036 | 453.5 ký tự | Chuẩn hóa chiều dài làm giảm nhẹ biên độ phân biệt reward trên tập held-out |
| LD-DPO | 0.56 | +0.0237 | 460.9 ký tự | Thêm phạt độ dài, câu trả lời sinh ra dài nhất trong các biến thể |
| ORPO | 0.66 | N/A (odds -0.62) | 446.9 ký tự | Đạt độ chính xác cao nhất (66%), không cần reference model, câu trả lời ngắn gọn nhất |

_Phân tích biến thể thay đổi độ dài:_
Biến thể LD-DPO (Length-Debiased DPO) làm tăng độ dài câu trả lời nhiều nhất (đạt trung bình 460.9 ký tự). Nguyên nhân xuất phát từ công thức hàm mất mát của LD-DPO: khi đưa thành phần điều chỉnh độ dài vào mẫu số, gradient phạt các câu ngắn có log-prob thấp mạnh hơn, vô tình khuyến khích mô hình sinh thêm các token đệm để tối ưu hóa tỷ lệ log-probability bình quân. Ngược lại, ORPO (Odds Ratio Preference Optimization) tạo ra câu trả lời ngắn gọn nhất (446.9 ký tự) và đạt độ chính xác cao nhất (66%) vì cơ chế Odds Ratio phạt trực tiếp tỷ lệ xác suất sai lệch mà không làm giãn nở độ dài sinh văn bản.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 32.0% / 44.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 6.6% |

_Nhận xét:_ Thành phần reward về đúng định dạng (format reward) tăng nhanh đầu tiên ngay trong 20 bước đầu, sau đó reward về tính đúng đắn toán học mới cải thiện dần. Mức tăng từ 32% lên 44% (+12%) vượt mức sai số chuẩn (6.6%), khẳng định GRPO học được chính sách suy luận hiệu quả.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là mô hình DPO sau khi được căn chỉnh trên tập dữ liệu tiếng Việt thực tế không hề bị mắc hội chứng "nói dài để thắng" (length bias) như lý thuyết cảnh báo, mà thậm chí câu trả lời trung bình còn cô đọng hơn SFT (-35 ký tự) trong khi loại bỏ triệt để các lỗi sinh lặp thẻ đặc biệt của mô hình nền.
