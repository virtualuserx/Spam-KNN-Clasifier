import nltk
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('stopwords')

from nltk.corpus import stopwords  # a, and, is gibi anlamsız kelime gruplarını kaldırmak#
from nltk.tokenize import word_tokenize # mesaji kelime parcalarina bolmek #
from nltk.stem import PorterStemmer  # do, does gibi ayni anlma gelen kelimeleri tek bir yapida yazmak #

import matplotlib.pyplot as plt  # gorsellik icin #
from wordcloud import WordCloud  # gorsellik icin, kelime veri kumesi olusturmak icin #

import os                      # elimdeki txt dosyalarini okumak icin #
import pandas as pd          # elimdeki txt dosyalarini okuduktan sonra DataFrame talo yapisina cevirmek icin #

from math import log    # yaptigin siniflandirma icin hesaplamarda kullandim #

#Fix: bilgisayarda nltk punkt,stopwords yoksa hata verebiliyor.

__name__= "__main__"
if __name__=="__main__":
    print("\n171180004 Bahattin Aksoy Uygulama Odevi2\n")

#################################################################################################

# verdigim linkte spam mailleri icin en iyi 3 algoritma arasinda knn oldugu icin proje knn uzerine kurulmustur
# link:https://www.matchilling.com/comparison-of-machine-learning-methods-in-email-spam-detection/

# knn icin yakınlık hesabini ise tf yontemi ile spamda gecen, hamda gecen kelimelere
#  ayrı ayrı agirlik hesaplari yapildiktan sonra bunları kayıt ettim
# train setindeki tum veriler icin eger bu veri spamda ise spam puanini kelimenin frekansına
# ve tüm mesajlarda geçme oranina gore puan verdim
# yeni dataframe icinde puan  ham ise 0, spam ise 1 degerini içermektedir.
# test verisi icinde bu hesabı yaptım ve ardindan eucledian tum komsular icin farkları bir dataframe icerisinde tuttum
# siraladiktan sonra belirtilen k degeri kadar komsudan en fazla sayiya sahip sinifi test icin cevap sundum
# en son dogru cevaplar ile karsilastirarak dogruluk degerini yazdirdim


# yaptigim testlerde  knn degerine bagli olarak farklı fakat yuzde 92 uzeri accuracy sonuclarina ulastim
# hem programi yormayacak hem de daha dogru siniflandirma icin k_value 7 olarak belirledim
# train data sabit tutarak interneten buldugum 747 test verisi ile %80'e yakin accuracy elde edebildim

###################################################################################################

# train verileri txt olarak okumak ve hepsini 1 dataframe icerisine aktarmak
print('veriler gereksiz kelimelrden arindiriliyor, duzenleniyor...')
print('veriler DataFrame icerisne aktariliyor...')

def process_message1(message):
    message = message.lower()
    stopWords = set(stopwords.words('english'))
    stopWords.add('subject')
    words = word_tokenize(message)
    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    stemmer = PorterStemmer()
    wordsFiltered = [stemmer.stem(word) for word in wordsFiltered]
    str = ""
    b = 0
    for i in range(len(wordsFiltered) - 1):
        str += wordsFiltered[i] + ' '
        b += 1
    str += wordsFiltered[b]
    return str

mymails = pd.DataFrame(columns=['label', 'message'])

directory = os.path.normpath("yz/training/ham")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f = open(os.path.join(subdir, file), 'r')
            text = f.read()
            text=process_message1(text)
            new_row = {'label': 0, 'message': text}
            mymails = mymails.append(new_row, ignore_index=True)
            f.close()

directory = os.path.normpath("yz/training/spam")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_message1(text)
            new_row = {'label': 1, 'message': text}
            mymails = mymails.append(new_row, ignore_index=True)
            f.close()

###################################################################################
# test verileri txt olarak okumak ve hepsini 1 dataframe icerisine aktarmak

testmails= pd.DataFrame(columns=['label', 'message'])

directory = os.path.normpath("yz/development/ham")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_message1(text)
            new_row = {'label': 0, 'message': text}
            testmails = testmails.append(new_row, ignore_index=True)
            f.close()

directory = os.path.normpath("yz/development/spam")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_message1(text)
            new_row = {'label': 1, 'message': text}
            testmails = testmails.append(new_row, ignore_index=True)
            f.close()

"""directory = os.path.normpath("yz/development/unseen")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_message1(text)
            new_row = {'label': 1, 'message': text}
            testmails = testmails.append(new_row, ignore_index=True)
            f.close()"""

trainData = mymails
testData = testmails

directory = os.path.normpath("yz/development/spam")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_message1(text)
            new_row = {'label': 1, 'message': text}
            testmails = testmails.append(new_row, ignore_index=True)
            f.close()




########################################################################################################
print('spam kelimeleri yazdiriliyor...')
spam_words = ' '.join(list(trainData[trainData['label'] == 1]['message']))
spam_wc = WordCloud(width = 400,height = 400).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

print('ham kelimeleri yazdiriliyor...')
ham_words = ' '.join(list(trainData[trainData['label'] == 0]['message']))
ham_wc = WordCloud(width = 400,height = 400).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


class SpamClassifier(object):
    def __init__(self, trainData,k_value):
        self.mails, self.labels = trainData['message'], trainData['label']
        self.k_value=k_value
        self.train()


    def train(self):
        print('knn egitiliyor...')
        print('kelimeleri frekanslari ve tüm mesajlarda geceme olasiliklari hesaplaniyor....')
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()
        print('tum training verileri hesaplamalara gore puanlandiriliyor....')
        self.create_neighbour_puan()


    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMessages):
            message_processed = word_tokenize(self.mails[i])
            count = list()
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1


    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                              / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                            / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails


    def create_neighbour_puan(self):
        self.komsu_puanlar = pd.DataFrame(columns=['label', 'puan_spam', 'puan_ham'])
        noOfMessages = self.mails.shape[0]
        for i in range(noOfMessages):
            processed_message = word_tokenize(self.mails[i])
            puan = self.classify(processed_message)
            new_row = {'label': self.labels[i], 'puan_spam': puan[0], 'puan_ham': puan[1]}
            self.komsu_puanlar = self.komsu_puanlar.append(new_row, ignore_index=True)
        data_types_dict = {'label': int}
        self.komsu_puanlar = self.komsu_puanlar.astype(data_types_dict)

# puanlandırıcı fonksiyon
    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return [pSpam, pHam]


    def predict(self, testData):
        print('test verileri hesaplamalara gore puanlandiriliyor....')
        print('test verileri ile trainin gerileri arasında spam_puan ve')
        print('ham_puan arasındaki farklar eucledian ile hesaplanıyor....')
        print('verilen knn degerine gore komsuların cogunlugana bakiliyor....')
        print('siniflandiriliyor, tahminler listeleniyor....')
        print('bu asama 3 dk surmektedir....')

        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = word_tokenize(message)
            result[i] = int(self.knn(processed_message))
        return result


    def knn(self, message):
        testmessage_puan = self.classify(message)
        return self.eucledian_hesapla(testmessage_puan)


    def eucledian_hesapla(self, test_puan):
        puan_spam = self.komsu_puanlar['puan_spam'].values
        puan_ham = self.komsu_puanlar['puan_ham'].values
        labels =  self.komsu_puanlar['label'].values
        noOfMessages = labels.shape[0]
        euceledian_hesaplar = pd.DataFrame(columns=['label', 'eucledian'])

        for i in range(noOfMessages):
            eucledian = ((test_puan[0] - puan_spam[i])**2 + abs(test_puan[1] - puan_ham[i])**2)**0.5
            new_row = {'label': labels[i], 'eucledian': eucledian}
            euceledian_hesaplar = euceledian_hesaplar.append(new_row, ignore_index=True)

        sorted = euceledian_hesaplar.sort_values(by=['eucledian'], axis=0)[:self.k_value]['label'].values
        ham_score = 0
        spam_score = 0
        for i in sorted:
            if (i == 1.0):
                spam_score += 1
            else:
                ham_score += 1
        if spam_score > ham_score:
            return 1
        else:
            return 0


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    print('\n')
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


sc_knn = SpamClassifier(trainData,7)
preds_knn = sc_knn.predict(testData['message'])
metrics(testData['label'], preds_knn)

"""
#pm=''' TESTING MESSAGE HERE    '''
if sc_knn.knn(pm)==1:
    print('This is a spam')
else:
    print('This is not Spam')
"""