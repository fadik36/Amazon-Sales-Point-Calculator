#####################################################
# Rating Product Sorting Rewiewsin Amazon
#####################################################
# İş Problemi
#####################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir.Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde
# sıralanması olarak karşımıza çıkmaktadır.Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem
# maddi kayıp hemde müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını
# arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

# Değişkenler
# region
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı

# helpful: Faydalı değerlendirme derecesi
# review: Text Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti

# unix Review Time: Değerlendirme zamanı
# review Time: Değerlendirme zamanı

# Rawday_diff: Değerlendirmeden itibaren geçen gün sayısı

# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı
# endregion

###############################################################
# Veriyi Anlama (Data Understanding)
###############################################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("week5/project_1_amazon_review_/amazon_review.csv")
df = df_.copy()  # kopyasını oluşturduk.
df.head()
df.shape  # (4915, 12)
df.isnull().sum()  # Sadece reviewerName ile reviewText değişkenlerinde 1'er tane eksik değer var.
df["reviewerName"].fillna(df["reviewerName"].mode()[0], inplace=True)  # eksik değerleri en çok tekrar eden değerle (mod) doldur.
df["reviewText"].fillna(df["reviewText"].mode()[0], inplace=True)

#################################
# Proje Görevleri
#################################

#########################################################################################################
# Görev 1:Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
#########################################################################################################
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız
# verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

# Adım 1: Ürünün ortalama puanını hesaplayınız.
df["overall"].head()
df["overall"].sum()  # 22548.0
# Ortalama Puan
df["overall"].mean()  # 4.587589013224822

# rating'in dağılımına bakalım:
df.head()
df["overall"].value_counts()
df["helpful_yes"].value_counts()
df["helpful_yes"].sum()  # 6444
df["total_vote"].value_counts()
df["total_vote"].sum()  # 7478
df.groupby("total_vote").agg({"helpful_yes": ["sum", "mean"]})  # faydalı bulunan oy sayılarının total_vote'e göre dağılımı.

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
# •reviewTime değişkenini tarih değişkeni olarak tanıtmanız.
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
# •reviewTime'ın max değerini current_date olarak kabul etmeniz.
df["current_date"] = df["reviewTime"].max()
# •her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve
#   gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden
#   gelen değerlere göre ağırlıklandırma yapmanız gerekir. Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan
#   yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["days"] = (df["current_date"] - df["reviewTime"]).dt.days
df["days"].quantile(0.28)
df.head()
df["days"].mean()

df.loc[df["days"] <= df["days"].quantile(0.28), "overall"].mean() * 28 / 100 + \
df.loc[(df["days"] > df["days"].quantile(0.28)) & (
        df["days"] <= df["days"].quantile(0.43636)), "overall"].mean() * 26 / 100 + \
df.loc[(df["days"] > df["days"].quantile(0.436)) & (
        df["days"] <= df["days"].quantile(0.6703)), "overall"].mean() * 24 / 100 + \
df.loc[(df["days"] > df["days"].quantile(0.6703)), "overall"].mean() * 22 / 100  # ortalama 4.605036666981506 geldi.

df["overall"].mean()  # 4.587589013224822


# "overall"a göre aldığımız ortalama "days" değişkenlerinin herbirinin quantieler çapında belli aralıklara göre
# "overall"ını seçtiğimiz ve ağırlıklandırdığımız sonuç daha yüksek çıktı. Biz bu ağırlıklandırmayı ne kadar değiştirirsek
# ortalama da da değişiklikler görebileceğiz. Ama bunu fonksiyonelleştirsek değişimi gözlemlememiz daha kolay olur sanki.
# Çünkü ağırlıklandırmları ne yönde değiştirdiğimiz bize verinin ne yönde nasıl bir değişim gösterdiğiniz açıklayacak.

def time_based_weighted_average(df, q1=0.28, q2=0.43636, q3=0.6703, w1=25, w2=25, w3=25, w4=25):
    return df.loc[df["days"] <= df["days"].quantile(q1), "overall"].mean() * w1 / 100 + \
           df.loc[(df["days"] > df["days"].quantile(q1)) & (
                       df["days"] <= df["days"].quantile(q2)), "overall"].mean() * w2 / 100 + \
           df.loc[(df["days"] > df["days"].quantile(q2)) & (
                       df["days"] <= df["days"].quantile(q3)), "overall"].mean() * w3 / 100 + \
           df.loc[(df["days"] > df["days"].quantile(q3)), "overall"].mean() * w4 / 100


time_based_weighted_average(df)  # Şimdiki Ortalaması 4.597603489250404 geldi.
# Biraz oynama yapalım:
time_based_weighted_average(df, 0.36, 0.5, 0.78)  # sadece quantileri yukarı çekmek ortalamayı ciddi oranda azalttı =4.568398131097687

# Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
df.loc[df["days"] <= df["days"].quantile(0.28), "overall"].mean()  # ort=4.701157742402316
df.loc[(df["days"] > df["days"].quantile(0.28)) & (
            df["days"] <= df["days"].quantile(0.43636)), "overall"].mean()  # ort =4.6379084967320265
df.loc[(df["days"] > df["days"].quantile(0.43636)) & (
            df["days"] <= df["days"].quantile(0.6703)), "overall"].mean()  # ort=4.577989601386482
df.loc[(df["days"] > df["days"].quantile(0.6703)), "overall"].mean()  # ort=4.473358116480793
# Buradan çıkaracağım sonuç değerlendirme günlerinde geriye gittikçe ürün rating'i düşüyor. Tam tersi zamanla kullanıcılar
# ürünlerin hakkında daha yüksek oylar kullanmaya başladı diyebiliriz. Zamanla bir iyileşme görülüyor.

##################################################################################
# Görev 2:  Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
##################################################################################

# Adım 1:  helpful_no değişkenini üretiniz.
df.head()
df.shape

# •total_vote bir yoruma verilen toplam up-down sayısıdır.
df["total_vote"].value_counts()
# •up, helpful demektir.
df["helpful"].value_counts()
df["helpful_yes"].value_counts()
# •Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# •Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(20)

# Adım 2:  score_pos_neg_diff, score_average_ratingve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.


# •score_pos_neg_diff, score_average_ratingve wilson_lower_boundskorlarını hesaplayabilmek için score_pos_neg_diff,
#   score_average_ratingve wilson_lower_bound fonksiyonlarını tanımlayınız.
comments = pd.DataFrame()
comments["user_ID"] = df["reviewerID"]
df.head()


# •score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisindescore_pos_neg_diff ismiyle kaydediniz.
def score_pos_neg_diff(up, down):
    return up - down


df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])
df.head()
comments["score_pos_neg_diff"]= score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])


# •score_average_rating'agöre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    else:
        return up / (up + down)


df["score_average_ratingve"] = df.apply(lambda i: score_average_rating(i["helpful_yes"], i["helpful_no"]), axis=1)
df["score_average_ratingve"].value_counts()
comments["score_average_ratingve"] = df.apply(lambda i: score_average_rating(i["helpful_yes"], i["helpful_no"]), axis=1)


# •wilson_lower_bound'agöre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda i: wilson_lower_bound(i["helpful_yes"], i["helpful_no"]), axis=1)
comments["wilson_lower_bound"] = df.apply(lambda i: wilson_lower_bound(i["helpful_yes"], i["helpful_no"]), axis=1)
# Adım 3:  20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
# •wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
# •Sonuçları yorumlayınız.
comments.sort_values("wilson_lower_bound", ascending=False).head(20)
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Bu sıralamayı wilson_lower_bound değişkenine göre yaptık ve gerçekten de sonuçlar tutarlı. Mesel sıralanmış bu tabloya
# baktığımız zamna ilk sıradaki kullanıcını wlb derecesi en yüksek ve 0.95754'tür. Buna paralel olarak total_vote'sine kıyasla
# helpful_yes değişkeni en yakın olan da bu kullanıcıdır. Yani yaptığı yorumda sosyal ispat dediğimiz kavramı bu kullanıcıda görebiliyoruzç
# Çünkü doğru değerlendirmeler başka kullanıcıların yorum yapmadan sadece bu yorumu faydalı bulduğu için oy vermesi için yeterlidir.
# Adı Hyoun Kim "Faluzure" olan kullanıcı yorum kısmının özetinde Galaxy S4 & Galaxy Tab 4 ile alakalı bir takım yorumlarda bulunmuş.

# Başka bir kullanıcıya geçelim bu sefer wlb değeri düşük olan olsun. Mesela wbl değeri 0.56552 olan kullanıcı ürün için overall'a 5 vermiş.
# "Use Nothing Other Than the Best" yorumunda bulunmuş. Bu bir nevi üründen memnun olduğunu gösteren bir ifadedir. En son 777 gün
# önce yorum yapan kullanıcı 5 değerlendirmeyle verilen oydan 5 faydalı bulma yorum oyu almış. Bu duruma bakacak olursak
# aslında score_average_ratingve değeri bize 1 değerini döndürüyor. Bu isteniken durumsa neden wlb değeri düşük?
# Çünkü Biz sadece orana bakıyoruz. Bu yeterli değil. Çünkü bu ürün için bu zamana kadar 7478 yorumda bulunulmuş. Bu yorumları
# Belli parametreler altında veriyi düzenlememiz ve wlp içinde bernolli istatistiğinin bulunması verinin genel dağılımndan yola çıkılarak
# az verinin diğer yorumların aldığı değerlendirmeler ve faydalı olma konusunda çok geride kaldığından wlp değeri düşük geliyor.

