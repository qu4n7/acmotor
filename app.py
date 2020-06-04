from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    # keras.backend.clear_session()
    scaler = load('scaler.bin')
    model = load_model('model')
    inp = [x for x in request.form.values()]

    # list of source columns
    clmns = ['Масса Нетто', 'Срок службы по ГОСТ 27.002-2015', 'Класс безопасности','Категория сейсмостойкости',
    'Климатическое исполнение по ГОСТ 15150-69', 'Категория размещения по ГОСТ 15150-69',
    'Тип атмосферы на объекте применения по ГОСТ 15150-69', 'Максимальная габаритная длина',
    'Максимальная габаритная ширина', 'Максимальная габаритная высота', 'Номинальная мощность', 'Номинальное напряжение',
    'Номинальная частота вращения', 'Высота оси вращения', 'Номинальная частота', 'Число фаз', 'Номинальный ток',
    'Количество отверстий', 'Расстояние между центрами отверстий на лапах', 'Диаметр вала', 'Соединение фаз обмоток статора',
    'Максимальная частота вращения', 'Кратность начального пускового тока', 'Кратность начального пускового момента',
    'Полное сопротивление', 'Режим работы', 'Система охлаждения по ГОСТ 20459-87', 'КПД', 'Коэффициент мощности',
    'Степень защиты электродвигателя по ГОСТ 14254-2015', 'Вероятность безотказной работы по ГОСТ 27.002-2015',
    'Условия транспортирования по ГОСТ 15150-69', 'Срок сохраняемости', 'Наличие сертификата системы обязательной сертификации ОИТ']

    clmns_processed = ['Масса Нетто', 'Срок службы по ГОСТ 27.002-2015', 'Класс безопасности', 'Категория сейсмостойкости', 
    'Категория размещения по ГОСТ 15150-69', 'Тип атмосферы на объекте применения по ГОСТ 15150-69',
    'Максимальная габаритная длина', 'Максимальная габаритная ширина', 'Максимальная габаритная высота', 'Номинальная мощность',
    'Номинальное напряжение', 'Номинальная частота вращения', 'Высота оси вращения', 'Номинальная частота', 'Число фаз',
    'Номинальный ток', 'Количество отверстий', 'Расстояние между центрами отверстий на лапах', 'Диаметр вала',
    'Максимальная частота вращения', 'Кратность начального пускового тока', 'Кратность начального пускового момента', 
    'Полное сопротивление', 'Режим работы', 'КПД', 'Коэффициент мощности', 'Вероятность безотказной работы по ГОСТ 27.002-2015',
    'Срок сохраняемости', 'Климатическое исполнение по ГОСТ 15150-69_-', 'Климатическое исполнение по ГОСТ 15150-69_absent',
    'Климатическое исполнение по ГОСТ 15150-69_М', 'Климатическое исполнение по ГОСТ 15150-69_О', 
    'Климатическое исполнение по ГОСТ 15150-69_Т', 'Климатическое исполнение по ГОСТ 15150-69_ТВ', 
    'Климатическое исполнение по ГОСТ 15150-69_ТМ', 'Климатическое исполнение по ГОСТ 15150-69_У',
    'Климатическое исполнение по ГОСТ 15150-69_УХЛ', 'Соединение фаз обмоток статора_4', 'Соединение фаз обмоток статора_Y',
    'Соединение фаз обмоток статора_absent', 'Соединение фаз обмоток статора_Δ/Υ', 'Соединение фаз обмоток статора_Звезда',
    'Соединение фаз обмоток статора_Треугольник', 'Система охлаждения по ГОСТ 20459-87_1C411',
    'Система охлаждения по ГОСТ 20459-87_IC0141', 'Система охлаждения по ГОСТ 20459-87_IC0142',
    'Система охлаждения по ГОСТ 20459-87_IC0143', 'Система охлаждения по ГОСТ 20459-87_IC0144',
    'Система охлаждения по ГОСТ 20459-87_IC7A1W7 по ГОСТ Р МЭК 60034-6', 'Система охлаждения по ГОСТ 20459-87_ICW37A71',
    'Система охлаждения по ГОСТ 20459-87_ICW37A81', 'Система охлаждения по ГОСТ 20459-87_absent',
    'Степень защиты электродвигателя по ГОСТ 14254-2015_IP23', 'Степень защиты электродвигателя по ГОСТ 14254-2015_IP44',
    'Степень защиты электродвигателя по ГОСТ 14254-2015_IP54', 'Степень защиты электродвигателя по ГОСТ 14254-2015_IP55',
    'Степень защиты электродвигателя по ГОСТ 14254-2015_absent', 'Условия транспортирования по ГОСТ 15150-69_absent',
    'Условия транспортирования по ГОСТ 15150-69_ОЖ1', 'Условия транспортирования по ГОСТ 15150-69_ОЖ3',
    'Условия транспортирования по ГОСТ 15150-69_ОЖ4', 'Наличие сертификата системы обязательной сертификации ОИТ_absent',
    'Наличие сертификата системы обязательной сертификации ОИТ_На стадии оформления',
    'Наличие сертификата системы обязательной сертификации ОИТ_Не сертифицируется',
    'Наличие сертификата системы обязательной сертификации ОИТ_Сертифицировано']

    x_pred = pd.DataFrame(dict(zip(clmns, [[i] for i in inp])))
    print(x_pred)
    clm_ = 'Класс безопасности'
    for i in range(x_pred.shape[0]):
        try:
            x_pred.loc[x_pred.index == i, clm_] = int(x_pred.loc[x_pred.index == i, clm_].values)
        except (ValueError,TypeError):
            x_pred.loc[x_pred.index == i, clm_] = 0
    x_pred[clm_] = x_pred[clm_].astype('int')

    clm_ = 'Категория сейсмостойкости'
    my_dict = {np.nan: 0, '-': 0, 'II': 2, 'I':1, 'III':3}
    x_pred[clm_] = x_pred[clm_].replace(my_dict)

    clm_ = 'Климатическое исполнение по ГОСТ 15150-69'
    x_pred.loc[x_pred[clm_].isnull(), clm_] = 'absent'
    x_pred = pd.concat([x_pred,pd.get_dummies(x_pred[clm_], prefix=clm_)], axis=1)
    x_pred.drop(clm_, axis=1, inplace=True)

    clm_ = 'Категория размещения по ГОСТ 15150-69'
    for i in range(x_pred.shape[0]):
        try:
            x_pred.loc[x_pred.index == i, clm_] = int(x_pred.loc[x_pred.index == i, clm_].values)
        except (ValueError,TypeError):
            x_pred.loc[x_pred.index == i, clm_] = 0
    x_pred[clm_] = x_pred[clm_].astype('int')

    clm_ = 'Тип атмосферы на объекте применения по ГОСТ 15150-69'
    my_dict = {np.nan: 0, '-': 0, 'II': 2, 'I':1, 'III':3, 'IV':4}
    x_pred[clm_] = x_pred[clm_].replace(my_dict)

    clm_ = 'Соединение фаз обмоток статора'
    x_pred.loc[x_pred[clm_].isnull(), clm_] = 'absent'
    x_pred = pd.concat([x_pred,pd.get_dummies(x_pred[clm_], prefix=clm_)], axis=1)
    x_pred.drop(clm_, axis=1, inplace=True)

    clm_ = 'Режим работы'
    my_dict = {np.nan: 0, 'S1': 1, 'Продолжительный (S1)': 1, 'S2': 2, 'S3': 3, 'S4': 4}
    x_pred[clm_] = x_pred[clm_].replace(my_dict)

    clm_ = 'Система охлаждения по ГОСТ 20459-87'
    x_pred.loc[x_pred[clm_].isnull(), clm_] = 'absent'
    x_pred = pd.concat([x_pred,pd.get_dummies(x_pred[clm_], prefix=clm_)], axis=1)
    x_pred.drop(clm_, axis=1, inplace=True)

    clm_ = 'Степень защиты электродвигателя по ГОСТ 14254-2015'
    x_pred.loc[x_pred[clm_].isnull(), clm_] = 'absent'
    x_pred = pd.concat([x_pred,pd.get_dummies(x_pred[clm_], prefix=clm_)], axis=1)
    x_pred.drop(clm_, axis=1, inplace=True)

    clm_ = 'Условия транспортирования по ГОСТ 15150-69'
    x_pred.loc[x_pred[clm_].isnull(), clm_] = 'absent'
    x_pred = pd.concat([x_pred,pd.get_dummies(x_pred[clm_], prefix=clm_)], axis=1)
    x_pred.drop(clm_, axis=1, inplace=True)

    clm_ = 'Наличие сертификата системы обязательной сертификации ОИТ'
    x_pred.loc[x_pred[clm_].isnull(), clm_] = 'absent'
    x_pred = pd.concat([x_pred,pd.get_dummies(x_pred[clm_], prefix=clm_)], axis=1)
    x_pred.drop(clm_, axis=1, inplace=True)

    for clm in [i for i in clmns_processed if i not in x_pred.columns]:
        x_pred[clm] = 0

    x_pred = x_pred[clmns_processed].astype(float)

    x_pred_scaled = x_pred.copy()
    for i in range(x_pred.shape[1]):
        x_pred_scaled.iloc[:,i] = ( x_pred.iloc[:,i] - scaler.mean_[i] ) / ( scaler.var_[i]**0.5 )

    '''
    # drop the potential columns arrived from dummies
    for clm in [i for i in x_pred.columns if i not in clmns_processed]:
        x_pred = x_pred.drop(clm, axis=1)
    '''
    
    prediction = model.predict(x_pred_scaled)

    return render_template(
        'index.html', 
        prediction_text=['Стоимость составит {:.2f}'.format(float(prediction))]
    )

if __name__ == "__main__":
    app.run(threaded=False)
