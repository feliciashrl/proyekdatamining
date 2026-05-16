# Analisis Sentimen Ulasan Aplikasi Halodoc di Google Play Store Menggunakan Stochastic Gradient Descent dan TF-IDF

**Nama Mahasiswa**, Budi Prasetiyo

*Jurusan Ilmu Komputer/Informatika, Universitas Negeri Semarang*
*bprasetiyo@mail.unnes.ac.id*

---

## Abstract

The rapid growth of digital health applications in Indonesia has led to a significant increase in user-generated reviews on platforms such as Google Play Store. Analyzing these reviews automatically is critical to understanding user satisfaction and improving service quality. This study aims to perform sentiment analysis on 1,200 user reviews of the Halodoc application scraped from Google Play Store using the Stochastic Gradient Descent (SGD) classifier with TF-IDF feature extraction. Text preprocessing steps including cleaning, case folding, slang word normalization, tokenization, stopword removal, and stemming using the Sastrawi library were applied. Sentiment labeling was performed using the InSet lexicon combined with a custom health-domain lexicon. Results showed that 61.5% of reviews were negative (738), 35.5% were positive (426), and 3% were neutral (36). The SGD classifier achieved an accuracy of 84.98% for both unigram and bigram TF-IDF configurations. N-gram analysis revealed that the most frequent positive keywords were *dokter*, *konsultasi*, *bagus*, *bantu*, and *cepat*, while negative reviews were dominated by *konsultasi*, *sesi*, *kesalahan*, *lambat*, and *tidak*. These findings indicate that service speed and consultation session quality are the primary drivers of user dissatisfaction. This study provides actionable insights for Halodoc's product development team.

**Keywords:** sentiment analysis, Halodoc, Stochastic Gradient Descent, TF-IDF, Google Play Store, text mining

---

## Abstrak

Pertumbuhan pesat aplikasi kesehatan digital di Indonesia menyebabkan meningkatnya jumlah ulasan pengguna di platform seperti Google Play Store. Analisis ulasan secara otomatis sangat penting untuk memahami kepuasan pengguna dan meningkatkan kualitas layanan. Penelitian ini bertujuan melakukan analisis sentimen terhadap 1.200 ulasan pengguna aplikasi Halodoc yang dikumpulkan dari Google Play Store menggunakan klasifikasi Stochastic Gradient Descent (SGD) dengan ekstraksi fitur TF-IDF. Tahapan preprocessing teks meliputi cleaning, case folding, normalisasi kata slang, tokenisasi, penghapusan stopword, dan stemming menggunakan library Sastrawi. Pelabelan sentimen dilakukan menggunakan leksikon InSet yang dikombinasikan dengan leksikon khusus domain kesehatan. Hasil menunjukkan bahwa 61,5% ulasan bersentimen negatif (738 ulasan), 35,5% positif (426 ulasan), dan 3% netral (36 ulasan). Klasifikasi SGD mencapai akurasi 84,98% pada konfigurasi unigram maupun bigram TF-IDF. Analisis n-gram mengungkap bahwa kata kunci positif yang paling sering muncul adalah *dokter*, *konsultasi*, *bagus*, *bantu*, dan *cepat*, sedangkan ulasan negatif didominasi oleh *konsultasi*, *sesi*, *kesalahan*, *lambat*, dan *tidak*. Temuan ini menunjukkan bahwa kecepatan layanan dan kualitas sesi konsultasi merupakan faktor utama ketidakpuasan pengguna. Penelitian ini memberikan wawasan yang dapat ditindaklanjuti oleh tim pengembang produk Halodoc.

**Kata kunci:** analisis sentimen, Halodoc, Stochastic Gradient Descent, TF-IDF, Google Play Store, text mining

---

## 1. Pendahuluan

Transformasi digital di sektor kesehatan telah mendorong lahirnya berbagai aplikasi telemedicine yang memungkinkan masyarakat mengakses layanan medis secara daring. Di Indonesia, penetrasi internet yang terus meningkat disertai dengan perubahan perilaku masyarakat pasca pandemi COVID-19 menjadikan aplikasi kesehatan digital sebagai kebutuhan primer [1]. Google Play Store sebagai platform distribusi aplikasi Android terbesar mencatat jutaan ulasan pengguna setiap harinya, yang merupakan sumber data berharga untuk memahami persepsi dan kepuasan pengguna terhadap suatu produk [2]. Halodoc, sebagai salah satu aplikasi telemedicine terkemuka di Indonesia dengan lebih dari 20 juta pengguna, menjadi objek kajian yang relevan untuk menganalisis opini publik secara sistematis.

Analisis sentimen atau opinion mining merupakan cabang dari Natural Language Processing (NLP) yang berfokus pada identifikasi dan ekstraksi opini subjektif dari teks [3]. Dalam konteks ulasan aplikasi, analisis sentimen memungkinkan pengembang untuk secara otomatis mengidentifikasi aspek layanan yang perlu diperbaiki tanpa harus membaca ribuan ulasan secara manual [4]. Berbagai pendekatan telah digunakan untuk analisis sentimen ulasan aplikasi berbahasa Indonesia, mulai dari pendekatan berbasis leksikon hingga machine learning. Penelitian [5] menggunakan Support Vector Machine (SVM) pada ulasan aplikasi Grab dengan akurasi 84,2%, sementara [6] menerapkan Naive Bayes pada ulasan aplikasi CGV Cinemas dan mencapai akurasi 81,3%. Penelitian [7] membandingkan Random Forest, SVM, dan Naive Bayes pada ulasan TikTok dan menemukan bahwa Random Forest memberikan performa terbaik dengan akurasi 94,15%.

Stochastic Gradient Descent (SGD) merupakan algoritma optimasi yang banyak digunakan dalam klasifikasi teks karena kemampuannya menangani dataset berskala besar dengan efisiensi komputasi tinggi [8]. Berbeda dengan metode batch gradient descent, SGD memperbarui parameter model menggunakan satu sampel data pada setiap iterasi, sehingga sangat cocok untuk pemrosesan data teks berdimensi tinggi seperti representasi TF-IDF [9]. Penerapan SGD dalam analisis sentimen teks ulasan pengguna telah terbukti memberikan performa kompetitif dengan waktu pelatihan yang lebih singkat dibandingkan SVM konvensional [10]. Kombinasi SGD dengan TF-IDF—baik unigram (n=1) maupun bigram (n=2)—memungkinkan pembandingan granularitas fitur teks yang berbeda dalam menangkap konteks ulasan bahasa Indonesia.

Meskipun penelitian analisis sentimen ulasan aplikasi berbahasa Indonesia cukup banyak dilakukan, terdapat beberapa celah yang belum terisi. Pertama, sebagian besar penelitian berfokus pada aplikasi e-commerce, transportasi, atau hiburan, sedangkan kajian mendalam pada aplikasi kesehatan telemedicine masih terbatas [6,7]. Kedua, penggunaan algoritma SGD yang dikombinasikan dengan perbandingan fitur unigram dan bigram TF-IDF secara eksplisit pada ulasan aplikasi kesehatan berbahasa Indonesia belum banyak dilaporkan. Ketiga, analisis insight berbasis n-gram untuk mengidentifikasi topik keluhan dan pujian secara spesifik pada aplikasi Halodoc belum pernah dipublikasikan sebelumnya. Penelitian [5] menggunakan SVM pada aplikasi transportasi, [6] menggunakan Naive Bayes pada aplikasi hiburan, dan [7] membandingkan tiga metode pada aplikasi media sosial—namun ketiganya tidak mengeksplorasi domain kesehatan dengan SGD secara bersamaan [5,6,7].

Penelitian ini bertujuan untuk menganalisis sentimen 1.200 ulasan pengguna aplikasi Halodoc yang dikumpulkan dari Google Play Store Indonesia menggunakan metode Stochastic Gradient Descent dengan ekstraksi fitur TF-IDF unigram dan bigram. Data ulasan dikumpulkan menggunakan library *google-play-scraper* pada Python. Preprocessing teks dilakukan melalui tujuh tahapan menggunakan library Sastrawi dan NLTK. Pelabelan sentimen menggunakan InSet Lexicon yang diperkaya dengan leksikon domain kesehatan. Output penelitian berupa distribusi sentimen, visualisasi word cloud, analisis top-5 keyword per sentimen, perbandingan performa unigram vs bigram, serta rekomendasi peningkatan layanan berbasis data.

---

## 2. Metode Penelitian

### 2.1. Kerangka Penelitian

Penelitian ini mengikuti alur pipeline text mining yang terdiri dari empat tahap utama: (1) pengumpulan data, (2) preprocessing teks, (3) pelabelan sentimen, dan (4) klasifikasi dan evaluasi. Seluruh implementasi dilakukan menggunakan bahasa pemrograman Python di lingkungan Google Colaboratory.

```
[Flowchart]
Data Collection (Google Play Store)
        ↓
Text Preprocessing
(Cleaning → Case Folding → Slang Fix → Tokenizing → Stopword Removal → Stemming)
        ↓
Sentiment Labeling (InSet Lexicon + Custom Lexicon)
        ↓
Feature Extraction (TF-IDF Unigram & Bigram)
        ↓
SGD Classification (Train/Test Split 80:20)
        ↓
Evaluasi & Analisis (Accuracy, Precision, Recall, F1, Confusion Matrix, N-gram)
```

*Gambar 1. Flowchart alur penelitian analisis sentimen ulasan Halodoc*

### 2.2. Pengumpulan Data

Data dikumpulkan dari Google Play Store Indonesia menggunakan library *google-play-scraper* versi 1.2.7 pada Python. Scraping dilakukan pada Mei 2026 dengan target aplikasi Halodoc (ID: `com.linkdokter.halodoc.android`), filter bahasa Indonesia (`lang='id'`), negara Indonesia (`country='id'`), dan metode pengurutan Most Relevant. Total data yang berhasil dikumpulkan berjumlah **1.200 ulasan** dengan rentang waktu dari September 2018 hingga Mei 2026. Dataset memiliki 11 kolom termasuk `content` (isi ulasan), `score` (rating 1–5), `at` (tanggal), dan `appVersion`. Tiga ulasan tidak memiliki data balasan (*replyContent* null), namun kolom `content` lengkap pada seluruh data.

Detail pengumpulan data dirangkum pada Tabel 1 berikut.

*Tabel 1. Detail Pengumpulan Data*

| Atribut | Keterangan |
|---|---|
| Sumber | Google Play Store Indonesia |
| ID Aplikasi | com.linkdokter.halodoc.android |
| Tools | google-play-scraper v1.2.7 (Python) |
| Tanggal Scraping | Mei 2026 |
| Jumlah Data | 1.200 ulasan |
| Bahasa | Indonesia |
| Rentang Waktu | September 2018 – Mei 2026 |

### 2.3. Text Preprocessing

Preprocessing teks dilakukan dalam tujuh tahapan berurutan untuk membersihkan dan menormalkan data teks sebelum analisis. Tabel 2 menampilkan contoh hasil setiap tahapan pada satu sampel ulasan.

*Tabel 2. Contoh Hasil Preprocessing Teks (1 Sampel Ulasan)*

| Tahap | Hasil |
|---|---|
| 1. Original | *"Aplikasi ini sangat membantu masyarakat yang tidak bisa langsung ke RS fisik terdekat untuk konsultasi tentang masalah kesehatan..."* |
| 2. Cleaning | Hapus mention, hashtag, URL, angka, karakter khusus |
| 3. Case Folding | Konversi ke huruf kecil |
| 4. Slangword Fix | Normalisasi kata informal (gak→tidak, udah→sudah, dll.) |
| 5. Tokenizing | Pemecahan teks menjadi token kata |
| 6. Stopword Removal | Penghapusan kata tidak bermakna (dari, dan, ke, dll.) |
| 7. Stemming | `['bantu', 'masyarakat', 'langsung', 'rs', 'fisik', 'dekat', 'konsultasi', 'kesehatan', ...]` |

Proses **cleaning** menghapus mention (@), hashtag (#), URL, angka, dan karakter non-alfanumerik menggunakan ekspresi reguler. **Case folding** mengubah seluruh teks menjadi huruf kecil. **Normalisasi kata slang** dilakukan menggunakan kamus 50+ kata informal konteks kesehatan dan media sosial Indonesia (misalnya: *lemot→lambat*, *konsul→konsultasi*, *loading→memuat*). **Tokenisasi** menggunakan fungsi `word_tokenize` dari library NLTK. **Stopword removal** menggunakan daftar stopword NLTK bahasa Indonesia dan Inggris yang diperkaya dengan 16 kata domain-spesifik (*halodoc*, *aplikasi*, *halo*, *min*, dll.). **Stemming** menggunakan StemmerFactory dari library Sastrawi dengan tambahan *exception dictionary* untuk 11 kata domain kesehatan penting (*pelayanan*, *konsultasi*, *pembayaran*, dll.) agar tidak salah di-stem.

Potongan kode utama tahap stemming dengan exception dictionary disajikan sebagai berikut:

```python
exception_dict = {
    'pelayanan':'pelayanan', 'konsultasi':'konsultasi',
    'pembayaran':'pembayaran', 'kesehatan':'kesehatan', ...
}
def stemmingText(word_list):
    result = []
    for w in word_list:
        if w in exception_dict:
            result.append(exception_dict[w])
        else:
            result.append(stemmer.stem(w))
    return result
```

*Kode 1. Fungsi stemming dengan exception dictionary untuk domain kesehatan*

### 2.4. Pelabelan Sentimen

Pelabelan sentimen dilakukan menggunakan pendekatan berbasis leksikon. Kamus InSet (*Indonesian Sentiment Lexicon*) [11] yang berisi 9.068 kata dengan bobot positif dan negatif digunakan sebagai basis. Kamus ini diperkaya dengan 28 kata tambahan konteks domain kesehatan dan aplikasi (misalnya: *responsif*=+3, *memuaskan*=+3, *gagal*=-3, *gangguan*=-2). Skor sentimen dihitung sebagai jumlah bobot seluruh kata dalam teks; skor > 0 diklasifikasikan positif, < 0 negatif, dan = 0 netral.

### 2.5. Ekstraksi Fitur dan Klasifikasi

Fitur teks diekstraksi menggunakan Term Frequency-Inverse Document Frequency (TF-IDF) dengan dua konfigurasi: **unigram** (`ngram_range=(1,1)`) dan **bigram** (`ngram_range=(1,2)`), masing-masing dengan `max_features=5.000`. Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan stratified split (`stratify=y`, `random_state=42`). Klasifikasi dilakukan menggunakan `SGDClassifier` dari scikit-learn dengan fungsi kerugian `hinge` (ekuivalen SVM linear) dan `max_iter=200`. Evaluasi model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

---

## 3. Hasil dan Pembahasan

### 3.1. Distribusi Sentimen

Dari 1.200 ulasan yang diproses, hasil pelabelan sentimen menunjukkan dominasi sentimen negatif sebesar **61,5%** (738 ulasan), diikuti sentimen positif **35,5%** (426 ulasan), dan netral **3%** (36 ulasan). Distribusi ini divisualisasikan pada Gambar 2.

*[Gambar 2. Distribusi Sentimen Ulasan Aplikasi Halodoc di Google Play Store (n=1.200) — lihat file distribusi_sentimen.png]*

Dominasi sentimen negatif mengindikasikan bahwa pengguna yang merasa tidak puas cenderung lebih aktif memberikan ulasan dibandingkan pengguna yang puas. Hal ini konsisten dengan fenomena *negativity bias* dalam perilaku konsumen daring [2]. Dari perspektif Sistem Informasi, temuan ini merupakan sinyal penting bagi manajemen Halodoc bahwa terdapat gap yang signifikan antara ekspektasi pengguna dan pengalaman aktual layanan.

### 3.2. Analisis Word Cloud per Sentimen

Visualisasi word cloud pada Gambar 3a (positif) dan 3b (negatif) memberikan gambaran intuitif tentang kata-kata yang mendominasi masing-masing kelas sentimen.

*[Gambar 3a. Word Cloud Sentimen Positif — lihat file wordcloud_positif.png]*
*[Gambar 3b. Word Cloud Sentimen Negatif — lihat file wordcloud_negatif.png]*
*[Gambar 3c. Word Cloud Sentimen Netral — lihat file wordcloud_netral.png]*

Pada ulasan positif, kata *dokter*, *konsultasi*, *bagus*, *bantu*, dan *cepat* mendominasi, mencerminkan kepuasan pengguna terhadap kualitas dokter, kemudahan konsultasi, dan kecepatan layanan. Sebaliknya, ulasan negatif didominasi oleh *konsultasi*, *sesi*, *kesalahan*, *lambat*, dan *tidak*, yang mengindikasikan keluhan seputar gangguan teknis, sesi konsultasi yang bermasalah, dan respons layanan yang lambat. Menariknya, kata *konsultasi* muncul di kedua kelas sentimen, menunjukkan bahwa fitur konsultasi adalah inti layanan yang menjadi titik kepuasan sekaligus ketidakpuasan utama pengguna.

### 3.3. Analisis Top-5 Keyword: Unigram dan Bigram

Analisis frekuensi n-gram memberikan pemahaman lebih mendalam tentang konteks keluhan dan pujian pengguna. Tabel 3 merangkum top-5 unigram dan bigram untuk masing-masing kelas sentimen.

*Tabel 3. Top-5 Unigram dan Bigram per Kelas Sentimen*

| Rank | Unigram Positif | Unigram Negatif | Bigram Positif | Bigram Negatif |
|---|---|---|---|---|
| 1 | dokter | konsultasi | dokter bagus | sesi selesai |
| 2 | konsultasi | sesi | konsultasi mudah | tidak membantu |
| 3 | bagus | kesalahan | bantu sekali | sesi berakhir |
| 4 | bantu | lambat | pelayanan bagus | tidak selesai |
| 5 | cepat | tidak | cepat respon | tidak bisa |

*[Gambar 4a. Top-5 Unigram Sentimen Positif — lihat file unigram_positif.png]*
*[Gambar 4b. Top-5 Bigram Sentimen Positif — lihat file bigram_positif.png]*
*[Gambar 4c. Top-5 Unigram Sentimen Negatif — lihat file unigram_negatif.png]*
*[Gambar 4d. Top-5 Bigram Sentimen Negatif — lihat file bigram_negatif.png]*

Analisis bigram memberikan konteks yang lebih kaya dibandingkan unigram. Bigram positif seperti *dokter bagus* dan *cepat respon* mengonfirmasi kepuasan terhadap kompetensi dokter dan responsivitas layanan. Sementara bigram negatif seperti *sesi selesai*, *sesi berakhir*, dan *tidak selesai* secara konsisten mengacu pada masalah pemutusan sesi konsultasi yang tidak wajar—sebuah *pain point* kritis yang tidak terungkap hanya dari analisis unigram. Temuan ini menunjukkan bahwa bigram memberikan nilai informatif yang lebih tinggi untuk identifikasi topik keluhan spesifik.

### 3.4. Performa Klasifikasi SGD

Tabel 4 menyajikan perbandingan performa SGD Classifier pada konfigurasi unigram dan bigram TF-IDF.

*Tabel 4. Perbandingan Hasil Evaluasi SGD Classifier*

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| SGD + Unigram TF-IDF | 84,98% | 85% | 85% | 85% |
| SGD + Bigram TF-IDF | 84,98% | 85% | 85% | 85% |

*[Gambar 5. Confusion Matrix SGD Classifier — Unigram TF-IDF (kiri) vs Bigram TF-IDF (kanan) — lihat file confusion_matrix.png]*

Kedua konfigurasi menghasilkan akurasi identik sebesar **84,98%**, yang menunjukkan bahwa pada dataset ini penambahan fitur bigram tidak memberikan peningkatan akurasi. Hal ini kemungkinan disebabkan oleh dominasi kata tunggal yang sudah cukup diskriminatif dalam membedakan sentimen positif dan negatif pada konteks ulasan aplikasi. Hasil ini sebanding dengan penelitian sejenis: [5] melaporkan akurasi SVM 84,2% pada ulasan Grab, [6] melaporkan Naive Bayes 81,3% pada ulasan CGV, dan [12] melaporkan SGD 74% pada klasifikasi teks multi-kelas—menunjukkan bahwa performa SGD pada penelitian ini kompetitif dengan metode lain.

Tabel 5 membandingkan penelitian ini dengan penelitian terdahulu yang relevan.

*Tabel 5. Perbandingan dengan Penelitian Terdahulu (Proposed Method Table)*

| Penelitian | Objek | Metode | Dataset | Akurasi |
|---|---|---|---|---|
| [5] Hidayat et al. (2021) | Grab (Play Store) | SVM + TF-IDF | 1.000 ulasan | 84,2% |
| [6] Evaluasi CGV (2024) | CGV Cinemas (Play Store) | Naive Bayes + TF-IDF | 4.600 ulasan | 81,3% |
| [7] TikTok Sentiment (2024) | TikTok (Play Store) | Random Forest + TF-IDF | 2.000 ulasan | 94,15% |
| [12] SGD Text Classification (2022) | Multi-domain | SGD + TF-IDF | Beragam | 74% |
| **Penelitian ini** | **Halodoc (Play Store)** | **SGD + TF-IDF** | **1.200 ulasan** | **84,98%** |

Penelitian ini memberikan kontribusi baru berupa (1) penerapan SGD pada domain telemedicine berbahasa Indonesia, (2) perbandingan eksplisit unigram vs bigram, dan (3) analisis insight mendalam berbasis n-gram yang dapat langsung ditindaklanjuti oleh pengembang produk.

### 3.5. Insight dan Rekomendasi

Berdasarkan hasil analisis, terdapat beberapa rekomendasi strategis untuk Halodoc. Pertama, **masalah teknis sesi konsultasi** (bigram: *sesi selesai*, *sesi berakhir*) harus diprioritaskan—keluhan ini mengindikasikan bug atau batas waktu sesi yang tidak sesuai ekspektasi pengguna dan berdampak langsung pada kepuasan. Kedua, **kecepatan respons dokter** perlu dipertahankan karena *cepat respon* muncul sebagai bigram positif tertinggi. Ketiga, **edukasi pengguna** tentang mekanisme sesi konsultasi perlu ditingkatkan untuk mengurangi miskomunikasi yang berujung pada ulasan negatif.

---

## 4. Kesimpulan

Penelitian ini berhasil melakukan analisis sentimen terhadap 1.200 ulasan aplikasi Halodoc dari Google Play Store menggunakan algoritma Stochastic Gradient Descent (SGD) dengan ekstraksi fitur TF-IDF. Hasil menunjukkan dominasi sentimen negatif (61,5%) yang mengindikasikan gap antara ekspektasi pengguna dan kualitas layanan aktual. Klasifikasi SGD mencapai akurasi 84,98% pada konfigurasi unigram maupun bigram, setara dengan metode SVM yang lebih umum digunakan. Analisis n-gram mengungkap bahwa *pain point* utama pengguna adalah masalah pemutusan sesi konsultasi yang tidak normal, sementara kepuasan tertinggi berkaitan dengan kualitas dokter dan kecepatan respons. Perbandingan unigram vs bigram menunjukkan bahwa fitur unigram sudah cukup representatif untuk klasifikasi sentimen ulasan aplikasi berbahasa Indonesia pada dataset ini. Untuk penelitian selanjutnya, disarankan mengeksplorasi model deep learning berbasis IndoBERT atau BERT multilingual yang dapat menangkap konteks kalimat secara lebih holistik, serta memperluas analisis aspek (*aspect-based sentiment analysis*) untuk granularitas insight yang lebih tinggi.

---

## Daftar Rujukan

[1] A. R. Pratama and R. Sciortino, "Patients' Perceptions in the Era of Digital Health Revolution: A Survey on Telemedicine in Indonesia," *Journal of Health Informatics in Developing Countries*, vol. 16, no. 1, pp. 1–12, 2022. doi: 10.18329/09737758/2022/16102

[2] B. Liu, "Sentiment Analysis and Opinion Mining," *Synthesis Lectures on Human Language Technologies*, vol. 5, no. 1, pp. 1–167, 2012. doi: 10.2200/S00416ED1V01Y201204HLT016

[3] A. Medhat, A. Hassan, and H. Korashy, "Sentiment Analysis Algorithms and Applications: A Survey," *Ain Shams Engineering Journal*, vol. 5, no. 4, pp. 1093–1113, 2014. doi: 10.1016/j.asej.2014.04.011

[4] Y. Mäntylä, D. Graziotin, and M. Kuutila, "The Evolution of Sentiment Analysis—A Review of Research Topics, Venues, and Top Cited Papers," *Computer Science Review*, vol. 27, pp. 16–32, 2018. doi: 10.1016/j.cosrev.2017.10.002

[5] D. Hidayat, R. Nuraeni, and I. Korespondensi, "Analisis Sentimen pada Review Aplikasi Grab di Google Play Store Menggunakan Support Vector Machine," *Jurnal Informatika*, vol. 8, no. 2, pp. 102–111, 2021. doi: 10.31294/ji.v8i2.9681

[6] R. Andriani and M. Faisal, "Evaluasi Sentimen Ulasan Pengguna CGV Cinemas Indonesia Menggunakan Metode Naive Bayes dan Support Vector Machine," *Indonesian Journal of Software Engineering*, vol. 10, no. 1, 2024. doi: 10.31294/ijse.v10i1.27099

[7] F. A. Ramadhan, "Sentiment Analysis of Tiktok App Reviews on Google Play Using Several Machine Learning Methods," *International Journal of Global Operations Research*, vol. 5, no. 4, pp. 122–131, 2024. doi: 10.47194/ijgor.v5i4.343

[8] L. Bottou, "Stochastic Gradient Descent Tricks," in *Neural Networks: Tricks of the Trade*, 2nd ed., Berlin: Springer, 2012, pp. 421–436. doi: 10.1007/978-3-642-35289-8_25

[9] T. Zhang, "Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms," in *Proc. 21st ICML*, 2004, pp. 116. doi: 10.1145/1015330.1015332

[10] P. Sharma and N. Sharma, "Evaluation of ML-Based Sentiment Analysis Techniques with Stochastic Gradient Descent and Logistic Regression," in *Proc. FICTA 2020*, Singapore: Springer, 2021, pp. 173–183. doi: 10.1007/978-981-33-6393-9_17

[11] F. Koto and A. Rahimi, "InSet Lexicon: Evaluating the Suitability of Word Sentiment Lexicon for the Indonesian Language," in *Proc. IALP 2017*, 2017, pp. 84–88. doi: 10.1109/IALP.2017.8300574

[12] C. Kula, M. Bielaniewicz, and K. Walkowiak, "Machine Learning Pipeline for Multi-Class Text Classification," *Applied Sciences*, vol. 12, no. 14, p. 7090, 2022. doi: 10.3390/app12147090

[13] R. Socher et al., "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank," in *Proc. EMNLP 2013*, 2013, pp. 1631–1642.

[14] A. Pak and P. Paroubek, "Twitter as a Corpus for Sentiment Analysis and Opinion Mining," in *Proc. LREC 2010*, 2010, pp. 1320–1326.

[15] I. Rish, "An Empirical Study of the Naive Bayes Classifier," in *Proc. IJCAI Workshop on Empirical Methods in AI*, 2001, pp. 41–46.

[16] V. N. Vapnik, *The Nature of Statistical Learning Theory*, 2nd ed., New York: Springer, 2000. doi: 10.1007/978-1-4757-3264-1

[17] A. L. Maas et al., "Learning Word Vectors for Sentiment Analysis," in *Proc. ACL 2011*, 2011, pp. 142–150.
