from transformers import pipeline
import torch # Eğer GPU kullanıyorsanız

# Özetleme pipeline'ını yükleyin. Model adını seçiminize göre değiştirin.
# Türkçe model örneği: model="dbmdz/mt5-base-turkish-summarization"
# İngilizce model örneği: model="facebook/bart-large-cnn"
# Cihazı belirtin (GPU varsa 'cuda', yoksa 'cpu')
device = 0 if torch.cuda.is_available() else -1 # device=0 ilk GPU'yu kullanır, device=-1 CPU'yu kullanır
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Özetlenecek metin (Türkçe bir model kullanıyorsanız Türkçe metin girin)
TEXT_TO_SUMMARIZE = """

"The strike hit right in the heart of the city. On Palm Sunday," Ukrainian President Volodymyr Zelensky says in an address following the Sumy strikes.

He says that among those left wounded by the Russian attack is a baby girl who was "born this year".

Zelensky adds that she was injured in the second of the two strikes that "exploded over the street".

Two children were killed in the Russian attacks, and 15 others were injured.

"Only completely deranged scum can do something like this," Zelensky says."""

# Özetleme işlemini yapın
# max_length: Özetin olabileceği maksimum token sayısı
# min_length: Özetin olması gereken minimum token sayısı
# do_sample=False: Daha deterministik (tutarlı) sonuçlar için genellikle False bırakılır.
summary = summarizer(TEXT_TO_SUMMARIZE, max_length=150, min_length=40, do_sample=False)

# Oluşturulan özeti yazdırın
print("----- ÖZET -----")
print(summary[0]['summary_text'])