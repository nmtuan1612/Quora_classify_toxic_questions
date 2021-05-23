import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import nltk
import csv
from PIL import Image
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import keras

# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
# from keras.layers import Bidirectional, GlobalMaxPool1D
# from keras.models import Model
# from keras import initializers, regularizers, constraints, optimizers, layers


# Đọc vào dữ liệu train và test

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# thử in ra vài dòng đầu của tập train
print(test_data.head())
#
# # biểu diễn tập train ở dạng biểu đồ cột
# sns.countplot(x='target', data=train_data)
# # plt.show()
#
# # kiểm tra xem tập train có chứa null values hoặc duplicates không
# print("Total Missing values in the Dataset :", train_data.isnull().sum().sum())
# print("Total Duplicated values in the Dataset :", train_data.duplicated().sum())
#
#
# # Data Preprocessing
#     #Feature Enginerring
#
#         # Đầu tiên ta chuyển đổi tập dữ liệu thô ban đầu thành tập các thuộc tính (features)
#         # để giúp biểu diễn tập dữ liệu ban đầu tốt hơn
#
# # Đối với tập train
# # độ dài câu hỏi
# train_data['qlen'] = train_data['question_text'].str.len()
#
# # số lượng từ trong câu hỏi
# train_data['n_words'] = train_data['question_text'].apply(lambda row: len(row.split(" ")))
#
# # số lượng từ dạng số trong câu hỏi
# train_data['numeric_words'] = train_data['question_text'].apply(lambda row: sum(c.isdigit() for c in row))
#
# # số lượng ký tự đặc biệt trong câu hỏi
# train_data['sp_char_words'] = train_data['question_text'].str.findall(r'[^a-zA-Z0–9 ]').str.len()
#
# # số lượng ký tự trong câu hỏi
# train_data['char_words'] = train_data['question_text'].apply(lambda row: len(str(row)))
#
# # số lượng từ duy nhất trong câu hỏi
# train_data['unique_words'] = train_data['question_text'].apply(lambda row: len(set(str(row).split())))
#
# # thử in ra vài dòng đầu để kiểm tra
# # print(train_data.head())
#
#
# # Đối với tập test
# # độ dài câu hỏi
# test_data['qlen'] = test_data['question_text'].str.len()
#
# # số lượng từ trong câu hỏi
# test_data['n_words'] = test_data['question_text'].apply(lambda row: len(row.split(" ")))
#
# # số lượng từ dạng số trong câu hỏi
# test_data['numeric_words'] = test_data['question_text'].apply(lambda row: sum(c.isdigit() for c in row))
#
# # số lượng ký tự đặc biệt trong câu hỏi
# test_data['sp_char_words'] = test_data['question_text'].str.findall(r'[^a-zA-Z0–9 ]').str.len()
#
# # số lượng ký tự trong câu hỏi
# test_data['char_words'] = test_data['question_text'].apply(lambda row: len(str(row)))
#
# # số lượng từ duy nhất trong câu hỏi
# test_data['unique_words'] = test_data['question_text'].apply(lambda row: len(set(str(row).split())))
#
# # thử in ra vài dòng đầu để kiểm tra
# # print(test_data.head())
#
#
# #Loại bỏ các nhiễu
#
# # Loại bỏ các dấu và ký tự đặc biệt trong câu
# # Loại bỏ các số trong câu
# # Sửa các từ sai chính tả thường gặp
# # Loại bỏ các từ viết tắt
# # Loại bỏ các từ không ảnh hưởng đến nghĩa của câu (stopword)
#
# # Loại bỏ các dấu và ký tự đặc biệt trong câu
# # Một danh sách đầy đủ các dấu và ký tự đặc biết được liệt kê trong mảng puncts
# puncts=[',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\',
#         '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
#         '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓',
#         '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '⋅', '‘', '∞',
#         '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√', '◄', '━',
#         '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～', '！', '○',
#         '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼', '☻', '┐',
#         '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰', '\x97', '↺',
#         '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗', '┗', '＊',
#         '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯', '☞', '´',
#         '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93', '≧', '］',
#         '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈', '％',
#         '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉', '☭',
#         '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥', '❝', '☐',
#         '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ', '❒', '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗', '܂', '☜',
#         '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣', '≪', '｢',
#         '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '＃', '⎯', '↠', '۩', '☰', '◥',
#         '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏', 'ⓐ',
#         '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂', '￦',
#         '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕', 'ⓝ',
#         '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃', '⋰', '♋',
#         '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘', '♞',
#         '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝', 'ⓑ',
#         '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤', '⬆', '⋱',
#         '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', '➣', '▿', 'ⓑ', '♉', '⏠', '◾', '▹',
#         '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪', '⊚',
#         '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉', '؛',
#         '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂', '␙', 'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽', '╘',
#         '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟', '⎛',
#         '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬', '⚑',
#         '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙', 'ⓦ',
#         '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕', '➘',
#         '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ', '⇛', '▊',
#         '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮', '☷',
#         '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎', '⇦', '␝',
#         '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲', '⩵', '̗', '❢',
#         '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢']
# def clean_punct(question):
#     for punct in puncts:
#         if punct in question:
#             question = question.replace(punct, ' ')
#     return question
#
# #Loại bỏ các số trong câu
# def clean_numbers(question):
#     # question = re.sub(r'[^a-z0-9A-Z]', " ", question)
#     if bool(re.search(r'\d', question)):  # xác định xem trong câu có chứa số hay không
#         question = re.sub('[0-9]{5,}', '#####', question) #Loại bỏ các số,thay bằng các '#'
#         question = re.sub('[0-9]{4}', '####', question)
#         question = re.sub('[0-9]{3}', '###', question)
#         question = re.sub('[0-9]{2}', '##', question)
#         question = re.sub('[0-9]{1}', '#', question)
#     return question
#
# #Sửa các từ sai chính tả thường gặp
# mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'bitcoin', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization',
#                 'electroneum':'bitcoin','nanodegree':'degree','hotstar':'star','dream11':'dream','ftre':'fire','tensorflow':'framework','unocoin':'bitcoin',
#                 'lnmiit':'limit','unacademy':'academy','altcoin':'bitcoin','altcoins':'bitcoin','litecoin':'bitcoin','coinbase':'bitcoin','cryptocurency':'cryptocurrency',
#                 'simpliv':'simple','quoras':'quora','schizoids':'psychopath','remainers':'remainder','twinflame':'soulmate','quorans':'quora','brexit':'demonetized',
#                 'iiest':'institute','dceu':'comics','pessat':'exam','uceed':'college','bhakts':'devotee','boruto':'anime',
#                 'cryptocoin':'bitcoin','blockchains':'blockchain','fiancee':'fiance','redmi':'smartphone','oneplus':'smartphone','qoura':'quora','deepmind':'framework','ryzen':'cpu','whattsapp':'whatsapp',
#                 'undertale':'adventure','zenfone':'smartphone','cryptocurencies':'cryptocurrencies','koinex':'bitcoin','zebpay':'bitcoin','binance':'bitcoin','whtsapp':'whatsapp',
#                 'reactjs':'framework','bittrex':'bitcoin','bitconnect':'bitcoin','bitfinex':'bitcoin','yourquote':'your quote','whyis':'why is','jiophone':'smartphone','dogecoin':'bitcoin','onecoin':'bitcoin','poloniex':'bitcoin','7700k':'cpu','angular2':'framework','segwit2x':'bitcoin','hashflare':'bitcoin','940mx':'gpu',
#                 'openai':'framework','hashflare':'bitcoin','1050ti':'gpu','nearbuy':'near buy','freebitco':'bitcoin','antminer':'bitcoin','filecoin':'bitcoin','whatapp':'whatsapp',
#                 'empowr':'empower','1080ti':'gpu','crytocurrency':'cryptocurrency','8700k':'cpu','whatsaap':'whatsapp','g4560':'cpu','payymoney':'pay money',
#                 'fuckboys':'fuck boys','intenship':'internship','zcash':'bitcoin','demonatisation':'demonetization','narcicist':'narcissist','mastuburation':'masturbation',
#                 'trignometric':'trigonometric','cryptocurreny':'cryptocurrency','howdid':'how did','crytocurrencies':'cryptocurrencies','phycopath':'psychopath',
#                 'bytecoin':'bitcoin','possesiveness':'possessiveness','scollege':'college','humanties':'humanities','altacoin':'bitcoin','demonitised':'demonetized',
#                 'brasília':'brazilia','accolite':'accolyte','econimics':'economics','varrier':'warrier','quroa':'quora','statergy':'strategy','langague':'language',
#                 'splatoon':'game','7600k':'cpu','gate2018':'gate 2018','in2018':'in 2018','narcassist':'narcissist','jiocoin':'bitcoin','hnlu':'hulu','7300hq':'cpu',
#                 'weatern':'western','interledger':'blockchain','deplation':'deflation', 'cryptocurrencies':'cryptocurrency', 'bitcoin':'blockchain cryptocurrency',}
#
# def _get_mispell(mispell_dict):
#     mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
#     return mispell_dict, mispell_re
#
# mispellings, mispellings_re = _get_mispell(mispell_dict)
# def replace_typical_misspell(text):
#     def replace(match):
#         return mispellings[match.group(0)]
#     return mispellings_re.sub(replace, text)
#
# #Loại bỏ các từ viết tắt
# contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
#                     "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
#                     "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
#                     "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would",
#                     "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have",
#                     "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have",
#                     "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
#                     "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
#                     "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
#                     "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
#                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
#                     "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
#                     "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
#                     "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
#                     "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
#                     "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
#                     "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
#                     "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
#                     "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
#                     "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
#                     "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
#
# def _get_contractions(contraction_dict):
#     contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
#     return contraction_dict, contraction_re
#
# contractions, contractions_re = _get_contractions(contraction_dict)
#
# def replace_contractions(text):
#     def replace(match):
#         return contractions[match.group(0)]
#     return contractions_re.sub(replace, text)
#
# #
# def to_lower(question):
#     return question.lower()
#
# # Hàm này được sử dụng để gọi tất cả các hàm tiền xử lý mà chúng ta đã xác định trước đó trên question_text.
# def clean_sentence(question):
#     question = clean_punct(question)
#     question = clean_numbers(question)
#     question = replace_typical_misspell(question)
#     question = replace_contractions(question)
# #     question = remove_stopwords(question)
#     question = to_lower(question)
#     question = question.replace("'", "")
#     return question
#
# # Tiền xử lý dữ liệu của tập train
# train_data['preprocessed_question_text'] = train_data['question_text'].apply(lambda x: clean_sentence(x))
# # print('train_quiz')
# # print(train_data.question_text.head())
# # print('preprocessed_train_question')
# # print(train_data.preprocessed_question_text.head())
#
# # Tiền xử lý dữ liệu của tập test
# test_data['preprocessed_question_text'] = test_data['question_text'].apply(lambda x: clean_sentence(x))
# # print('test_quiz')
# # print(test_data.question_text.head())
# # print('preprocessed_test_question')
# # print(test_data.preprocessed_question_text.head())
#
# #Chia tập train và tập validation
# # random 90% làm mảng train, 10% làm mảng validation
# train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=123)
#
# ## some config values
# embed_size = 300 # kích thước một word vector
# max_features = 50000 # how many unique words to use (i.e num rows in emedding vector)
# maxlen = 100 # max number of words in a question to use
#
# ## lấp đầy các missing values
# train_X = train_data["preprocessed_question_text"].fillna("_na_").values
# val_X = val_data["preprocessed_question_text"].fillna("_na_").values
# test_X = test_data["preprocessed_question_text"].fillna("_na_").values
#
# ## sử dụng thuật toán tách từ và chuyển đổi chúng thành chuỗi vector
# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(train_X))
# train_X = tokenizer.texts_to_sequences(train_X)
# val_X = tokenizer.texts_to_sequences(val_X)
# test_X = tokenizer.texts_to_sequences(test_X)
#
# print(train_X)

