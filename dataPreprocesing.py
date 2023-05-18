import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#! Veri Okuma
def read_data():
    covidData = pd.read_csv("Covid Dataset.csv")
    df = covidData.copy()
    return df


#! Veri özellikleri
def data_info(df,info="Default"):
    print("***************Data Information For", info,"*****************")
    print("Covid Verileri \n", df)
    print("Veri kolonları \n", df.columns)
    print("Veri Bilgisi \n", df.info())
    print("Veri Açıklamaları \n", df.describe().T)
    print("Veri NaN Değerleri Kontrol\n", df.isnull().sum())


# Veriler de eksik değer olmadığı için eksik değer cıkarma/ doldurma işlemi yapılmadı

#! Veri Özellikleri Görselleştirme
def data_vizulation(df):
    #?covıd vakası olanların/ olmayanların gorseli
    sns.countplot(x="COVID-19", data=df)
    plt.title("Covid Olanlar ve Olmayanlar")
    plt.show()

    #? covıd vakası olan(yes) olmayanların(no) yüzdelik dilimi
    df["COVID-19"].value_counts().plot.pie(explode=[0.1,0.1],autopct="%1.1f%%",shadow=True)
    plt.title("Yüzdelik Oran ")
    plt.title("Covid Olanların ve Olmayanların Yüzdelik Oranı ")
    plt.show()

    #? Solunum problemeninin evet/hayır da etkileri
    sns.countplot(x="Breathing Problem", hue="COVID-19",data=df)
    plt.title("Solunum Problemi ")
    plt.show()
    #? Ateş problemeninin evet/hayır da etkileri
    sns.countplot(x="Fever", hue="COVID-19", data=df)
    plt.title("Ateş Problemi")
    plt.show()

    #? Kuru Öksürük evet/hayır da etkileri
    sns.countplot(x="Dry Cough", hue="COVID-19", data=df)
    plt.title("Kuru Öksürük Problemi")
    plt.show()

    #? Boğaz Ağrısının evet/hayır da etkileri
    sns.countplot(x="Sore throat", hue="COVID-19", data=df)
    plt.title("Boğaz Ağrısı Problemi")
    plt.show()

    #? Burun Akıntısının evet/hayır da etkileri
    sns.countplot(x="Running Nose", hue="COVID-19", data=df)
    plt.title("Burun Akıntısı Problemi")
    plt.show()

    #? Astım probleminin evet/hayır da etkileri
    sns.countplot(x="Asthma", hue="COVID-19", data=df)
    plt.title("Astım Problemi")
    plt.show()

    #? Diabetin evet/hayır da etkileri
    sns.countplot(x="Diabetes", hue="COVID-19", data=df)
    plt.title("Diabet  Problemi")
    plt.show()

    #? Maskenin evet/hayır da etkileri
    sns.countplot(x="Wearing Masks", hue="COVID-19", data=df)
    plt.title("Maske Kullanımı  ")
    plt.show()

from sklearn.preprocessing import LabelEncoder
def encoder_features(df):
    encode=LabelEncoder()
    #? Breathing Problem encoder
    df["Breathing Problem"]=encode.fit_transform(df["Breathing Problem"])
    #? Fever encoder
    df["Fever"]=encode.fit_transform(df["Fever"])
    #? Dry Cough encoder
    df["Dry Cough"]=encode.fit_transform(df["Dry Cough"])
    #? Sore throat encoder
    df["Sore throat"]=encode.fit_transform(df["Sore throat"])
    #? Running Nose encoder
    df["Running Nose"]=encode.fit_transform(df["Running Nose"])
    #? Asthma encoder
    df["Asthma"]=encode.fit_transform(df["Asthma"])
    #? Chronic Lung Disease Encoder
    df["Chronic Lung Disease"] = encode.fit_transform(df["Chronic Lung Disease"])
    #? Headache Encoder
    df["Headache"] =encode.fit_transform(df["Headache"])
    #? Heart Disease Encoder
    df["Heart Disease"] = encode.fit_transform(df["Heart Disease"])
    #? Diabetes Encoder
    df["Diabetes"] = encode.fit_transform(df["Diabetes"])
    #? Hyper Tension Encoder
    df["Hyper Tension"] = encode.fit_transform(df["Hyper Tension"])
    #? Fatigue Encoder
    df["Fatigue"] =encode.fit_transform(df["Fatigue"])
    #? Gastrointestinal Encoder
    df["Gastrointestinal"] = encode.fit_transform(df["Gastrointestinal"])
    #? Abroad travel Encoder
    df["Abroad travel"] = encode.fit_transform(df["Abroad travel"])
    #? Contact with COVID Patient Encoder
    df["Contact with COVID Patient"] =encode.fit_transform(df["Contact with COVID Patient"])
    #? Large Gathering Encoder
    df["Attended Large Gathering"] = encode.fit_transform(df["Attended Large Gathering"])
    #? Visited Public Exposed Places Encoder
    df["Visited Public Exposed Places"] = encode.fit_transform(df["Visited Public Exposed Places"])
    #? Family working in Public Exposed Places Encoder
    df["FamilyWorking_in_PublicExposedPlaces"] = encode.fit_transform(df["FamilyWorking_in_PublicExposedPlaces"])
    #? Wearing Masks Encoder
    df["Wearing Masks"] = encode.fit_transform(df["Wearing Masks"])
    #?  Sanitization from Market Encoder
    df["Sanitization from Market"] = encode.fit_transform(df["Sanitization from Market"])
    #? COVID-19
    df["COVID-19"] = encode.fit_transform(df["COVID-19"])

    # ? 0/1 data vizulation

    #df.hist(figsize=(20,15))
    #plt.title("All Features Information")
    #plt.show()

    return df

#! Feature Selection
def select_Feature(df):
    df = df.drop("Running Nose",axis=1)
    df = df.drop("Chronic Lung Disease", axis=1)
    df = df.drop("Headache", axis=1)
    df = df.drop("Fatigue", axis=1)
    df = df.drop("Gastrointestinal", axis=1)
    df = df.drop("Wearing Masks", axis=1)
    df = df.drop("Sanitization from Market", axis=1)

    # ? return df without drop-features
    return df

#%%
def corr_Table(df,name="Correlation Table"):
    # ? Correlation Table without drop Features
    fig, ax = plt.subplots(figsize=(20,20))
    correlationTable = df.corr()
    sns.heatmap(correlationTable, annot=True, annot_kws={'size': 12}, linewidths=.4, ax=ax)
    plt.title(name)
    #fig.savefig(name +"corrTable.png")
    #plt.savefig("corrTable.png")
    plt.show()
#%%





df=read_data()
data_info(df," First Upload Data")
#data_vizulation(df)
encoded_data=encoder_features(df)
#print(encoded_data.head(8))
#data_info(encoded_data,"Encoded Features")
#data_vizulation(encoded_data)
#! Not  : Wearing Masks and Sanitization from Market one values
#? Correlation Table called
corr_Table(df,"Encoded Features Correlation Table")
#! NOTE: Out features <<>>>>  Running Nose , Chronic Lung Disease, Headache, Fatigue
#! Gastrointestinal , Wearing Masks, Sanitization from Market

selected_features=select_Feature(encoded_data)
corr_Table(selected_features,"Selected Features Correlation Table")

def read_fakeData():
    fake_data=pd.read_csv("Covid-19_Dataset.csv")
    fake_data=encoder_features(fake_data)
    fake_data=select_Feature(fake_data)
    return fake_data

fake_data=read_fakeData()