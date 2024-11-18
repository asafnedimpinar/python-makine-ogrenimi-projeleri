import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

# Labirent oluşturuluyor
labirent = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -100, -100],
    [-100, -100, -1, -100, -100, -100, -100, -100, -100, -100, -1, -1, -1, -100, -100],
    [-100, -1, -1, -100, -1, -1, -1, -1, -1, -100, -1, -100, -1, -100, -100],
    [-100, -1, -100, -1, -1, -100, -100, -100, -1, -1, -1, -100, -1, -100, -100],
    [-100, -1, -100, -1, -100, -100, -1, -1, -1, -100, -100, -100, -1, -100, -100],
    [-100, -1, -100, -1, -100, -100, -1, -100, -1, -1, -1, -1, -1, -100, -100],
    [-100, -1, -100, -1, -100, -1, -1, -100, -100, -100, -100, -100, -1, -100, -100],
    [-100, -1, -100, -1, -100, -1, -1, -1, -100, -1, -1, -1, -1, -100, -100],
    [-100, -1, -100, -1, -1, -1, -100, -1, -1, -1, -100, -100, -1, -1, -100],
    [-100, -1, -1, -1, -100, -100, -100, -1, -100, -100, -100, -1, -1, -1, -100],
    [-100, -100, -100, -100, -1, -1, -1, -1, -1, -100, -100, -1, -100, -1, -100],
    [-1, -1, -1, -1, -1, -100, -1, -100, -1, -1, -1, -1, -100, -1, -100],
    [-100, -100, -100, -100, -100, -100, -1, -100, -1, -100, -100, -1, -1, -1, -100],
    [-100, -100, -100, -100, -100, -1, -1, -1, -1, -100, -1, -100, -100, -100, -100],
    [-100, -100, -100, -100, -100, -1, -100, -100, -100, -1, -1, -1, -1, -1, -1]
])

# Labirent satır ve sütun sayısını al
labirent_satır_sayısı, labirent_sütun_sayısı = labirent.shape

# Q-değerleri (state-action pair) için sıfırlardan oluşan bir matris oluştur
q_degerleri = np.zeros((labirent_satır_sayısı, labirent_sütun_sayısı, 4))

# Hareket yönleri (sağa, sola, yukarı, aşağı)
hareketler = ["SAG", "SOL", "YUKARI", "ASAGI"]

# Bir hücrede engel olup olmadığını kontrol eden fonksiyon
def engel_mi(gecerli_satir_index, gecerli_sütun_index):
    return labirent[gecerli_satir_index, gecerli_sütun_index] == -1

# Labirentte rastgele bir başlangıç pozisyonu belirleyen fonksiyon
def baslangıc_belirle():
    while True:
        gecerli_satir_index = np.random.randint(labirent_satır_sayısı)
        gecerli_sütun_index = np.random.randint(labirent_sütun_sayısı)
        if not engel_mi(gecerli_satir_index, gecerli_sütun_index):
            return gecerli_satir_index, gecerli_sütun_index

# Hareket yönünü belirleyen fonksiyon
def hareket_belirle(gecerli_satir_index, gecerli_sütun_index, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(q_degerleri[gecerli_satir_index, gecerli_sütun_index])
    else:
        return np.random.randint(4)
    
# Belirlenen yöne göre hareket eden fonksiyon
def hareket_et(gecerli_satir_index, gecerli_sütun_index, hareket_index):
    yeni_satır_index = gecerli_satir_index
    yeni_sütun_index = gecerli_sütun_index

    if hareketler[hareket_index] == "SAG" and gecerli_sütun_index < labirent_sütun_sayısı - 1:
        yeni_sütun_index += 1
    elif hareketler[hareket_index] == "SOL" and gecerli_sütun_index > 0:
        yeni_sütun_index -= 1
    elif hareketler[hareket_index] == "YUKARI" and gecerli_satir_index > 0:
        yeni_satır_index -= 1
    elif hareketler[hareket_index] == "ASAGI" and gecerli_satir_index < labirent_satır_sayısı - 1:
        yeni_satır_index += 1

    return yeni_satır_index, yeni_sütun_index

# Başlangıç pozisyonundan en kısa yolu bulan fonksiyon
def en_kısa_yol(bas_satır_index, baslangıc_sütun_indeks):
    if engel_mi(bas_satır_index, baslangıc_sütun_indeks):
        return []
    else:
        gecerli_satir_index, gecerli_sütun_index = bas_satır_index, baslangıc_sütun_indeks
        en_kısa = []
        en_kısa.append([gecerli_satir_index, gecerli_sütun_index])
        while not engel_mi(gecerli_satir_index, gecerli_sütun_index):
            hareket_index = hareket_belirle(gecerli_satir_index, gecerli_sütun_index, 1)
            gecerli_satir_index, gecerli_sütun_index = hareket_et(gecerli_satir_index, gecerli_sütun_index, hareket_index)
            en_kısa.append([gecerli_satir_index, gecerli_sütun_index])
        return en_kısa

# Q-learning algoritması
epsilon = 0.9
azalma_degeri = 0.9
ogrenme_oranı = 0.9

for adım in range(1000):
    satır_index, sütun_index = baslangıc_belirle()
    while not engel_mi(satır_index, sütun_index):
        hareket_index = hareket_belirle(satır_index, sütun_index, epsilon)
        eski_satır_index, eski_sütun_index = satır_index, sütun_index
        satır_index, sütun_index = hareket_et(satır_index, sütun_index, hareket_index)
        ödül = labirent[satır_index, sütun_index]
        eski_q_degeri = q_degerleri[eski_satır_index, eski_sütun_index, hareket_index]
        fark = ödül + (azalma_degeri * np.max(q_degerleri[satır_index, sütun_index])) - eski_q_degeri
        yeni_q_degeri = eski_q_degeri + (ogrenme_oranı * fark)
        q_degerleri[eski_satır_index, eski_sütun_index, hareket_index] = yeni_q_degeri

print("Eğitim tamamlandı") 

# Robotun başlangıç konumunu belirle ve en kısa yolu bul
while True:
    baslangıc_satır, baslangıc_sütun = map(int, input("Robotun konumunu girin (satır,sütun): ").split(','))
    en_kısa_rota = en_kısa_yol(baslangıc_satır, baslangıc_sütun)
    
    if not en_kısa_rota:
        print("Geçersiz konum. Lütfen tekrar deneyin.")
    else:
        print("Çıkışa giden rota:")
        for i in range(len(en_kısa_rota)):
            print(en_kısa_rota[i])
        break  # Geçerli bir konum bulunduğunda döngüden çık15,1

# Labirenti ve ajan yolunu çiz
def ciz_labirent_ajan_yol(labirent, ajan_konumu, başlangıc_konumu, yol):
    fig, ax = plt.subplots()
    ax.imshow(labirent, cmap="gray")

    for i in range(labirent.shape[0]):
        for j in range(labirent.shape[1]):
            if labirent[i, j] == -1:
                renk = "green" if (i, j) == başlangıc_konumu else "red"
                circle = plt.Circle((j, i), 0.3, color=renk)
                ax.add_patch(circle)
            elif labirent[i, j] == -100:
                ax.text(j, i, "X", ha="center", va="center", color="red", fontsize=8)
            elif labirent[i, j] == 100:
                ax.text(j, i, "Hedef", ha="center", va="center", color="green", fontsize=8)

    yol_np = np.array(yol)
    ax.plot(yol_np[:, 1], yol_np[:, 0], color="blue", linewidth=1)

    ax.plot(ajan_konumu[1], ajan_konumu[0], "bo", markersize=15)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_aspect("equal", adjustable="box")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

ciz_labirent_ajan_yol(labirent, en_kısa_rota[0], (baslangıc_satır, baslangıc_sütun), en_kısa_rota)

# Ajanı adım adım göster
for konum in en_kısa_rota[1:]:
    clear_output(wait=True)
    ciz_labirent_ajan_yol(labirent, konum, (baslangıc_satır, baslangıc_sütun), en_kısa_rota)
    time.sleep(0.05)