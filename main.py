from transformers import pipeline
import torch # Eğer GPU kullanıyorsanız

# Özetleme pipeline'ını yükleyin. Model adını seçiminize göre değiştirin.
# Türkçe model örneği: model="dbmdz/mt5-base-turkish-summarization"
# İngilizce model örneği: model="facebook/bart-large-cnn"
# Cihazı belirtin (GPU varsa 'cuda', yoksa 'cpu')
device = 0 if torch.cuda.is_available() else -1 # device=0 ilk GPU'yu kullanır, device=-1 CPU'yu kullanır
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Özetlenecek metin (Türkçe bir model kullanıyorsanız Türkçe metin girin)
TEXT_TO_SUMMARIZE = """Buraya özetlenecek metni girin."""

# Özetleme işlemini yapın
# max_length: Özetin olabileceği maksimum token sayısı
# min_length: Özetin olması gereken minimum token sayısı
# do_sample=False: Daha deterministik (tutarlı) sonuçlar için genellikle False bırakılır.
summary = summarizer(TEXT_TO_SUMMARIZE, max_length=150, min_length=40, do_sample=False)

# Oluşturulan özeti yazdırın
print("----- ÖZET -----")
print(summary[0]['summary_text'])