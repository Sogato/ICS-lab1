import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)


def analyze_dataset(file_path):
    df = pd.read_csv(file_path)

    # Определение столбцов для анализа
    columns_for_analysis = [
        'temp',
        'wd',
        'ws',
        'atmos_pres',
        'dew_point',
        'rh',
        'ceil_hgt',
        'visibility',
    ]
    df = delete_rows_with_missing_values(df, columns_for_analysis)

    description = df.describe()
    attributes = df.columns.tolist()
    missing_values = df.isnull().sum()

    # Вывод результатов
    print("Описание датасета:")
    print(description)
    print("\nСписок атрибутов:")
    print(attributes)
    print("\nПропущенные значения для каждого атрибута:")
    print(missing_values)


def delete_rows_with_missing_values(df, columns):
    # Удаление строк с пропущенными значениями только в указанных столбцах
    return df.dropna(subset=columns)


def display_heatmap(df, features):
    # Словарь с укороченными названиями для удобства отображения на графике
    short_names = {
        'temp': 'Температура',
        'wd': 'Направление ветра',
        'ws': 'Скорость ветра',
        'atmos_pres': 'Давление',
        'dew_point': 'Точка росы',
        'rh': 'Влажность',
        'ceil_hgt': 'Нижняя граница облака',
        'visibility': 'Видимость'
    }

    # Выборка данных по заданным признакам
    selected_data = df[features].rename(columns=short_names)

    # Вычисление матрицы корреляции
    correlation_matrix = selected_data.corr()

    # Отображение тепловой карты
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', fmt=".2f", linewidths=.5, edgecolor='gray')
    plt.title('Тепловая карта корреляции между признаками')
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.yticks(rotation=30)

    plt.savefig('graph/heatmap.png', dpi=200)


def triangular_membership_function(x, a, b, c):
    # Функция проверяет, в какой точке находится x и вычисляет функцию принадлежности
    if x < a or x > c:
        return 0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def define_triangular_terms(df, column):
    # Определение минимума и максимума для данной колонки
    min_value = df[column].min()
    max_value = df[column].max()
    # Создание термов на основе минимума и максимума
    range_value = max_value - min_value
    return [
        {'name': 'Очень низкий', 'a': min_value, 'b': min_value, 'c': min_value + range_value * 0.25},
        {'name': 'Низкий', 'a': min_value + range_value * 0.2, 'b': min_value + range_value * 0.25,
         'c': min_value + range_value * 0.5},
        {'name': 'Средний', 'a': min_value + range_value * 0.4, 'b': min_value + range_value * 0.5,
         'c': min_value + range_value * 0.6},
        {'name': 'Высокий', 'a': min_value + range_value * 0.5, 'b': min_value + range_value * 0.75,
         'c': min_value + range_value * 0.8},
        {'name': 'Очень высокий', 'a': min_value + range_value * 0.75, 'b': max_value, 'c': max_value}
    ]


def plot_triangular_mfs(df, column, terms):
    x_values = df[column].sort_values().unique()

    plt.figure(figsize=(10, 5))

    # Генерируем и отображаем функцию принадлежности для каждого терма
    for term in terms:
        y_values = [triangular_membership_function(x, term['a'], term['b'], term['c']) for x in x_values]
        plt.plot(x_values, y_values, label=term['name'])

    plt.title(f'Треугольные функции принадлежности для {column}')
    plt.xlabel('Значение переменной')
    plt.ylabel('Степень принадлежности')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'graph/plot_triangular/{column}_mf.png', dpi=200)


def trapezoidal_membership_function(x, a, b, c, d):
    if x < a or x > d:
        return 0
    elif a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    else:  # c < x <= d
        return (d - x) / (d - c)


def define_trapezoidal_terms(df, column):
    min_value = df[column].min()
    max_value = df[column].max()
    range_value = max_value - min_value
    return [
        {'name': 'Очень низкий', 'a': min_value, 'b': min_value, 'c': min_value + range_value * 0.1,
         'd': min_value + range_value * 0.3},
        {'name': 'Низкий', 'a': min_value + range_value * 0.2, 'b': min_value + range_value * 0.3,
         'c': min_value + range_value * 0.4, 'd': min_value + range_value * 0.5},
        {'name': 'Средний', 'a': min_value + range_value * 0.4, 'b': min_value + range_value * 0.45,
         'c': min_value + range_value * 0.55, 'd': min_value + range_value * 0.6},
        {'name': 'Высокий', 'a': min_value + range_value * 0.5, 'b': min_value + range_value * 0.6,
         'c': min_value + range_value * 0.7, 'd': min_value + range_value * 0.8},
        {'name': 'Очень высокий', 'a': min_value + range_value * 0.7, 'b': min_value + range_value * 0.8,
         'c': max_value, 'd': max_value}
    ]


def plot_trapezoidal_mfs(df, column, terms):
    x_values = df[column].sort_values().unique()

    plt.figure(figsize=(10, 5))

    for term in terms:
        y_values = [trapezoidal_membership_function(x, term['a'], term['b'], term['c'], term['d']) for x in x_values]
        plt.plot(x_values, y_values, label=term['name'])

    plt.title(f'Трапециевидные функции принадлежности для {column}')
    plt.xlabel('Значение переменной')
    plt.ylabel('Степень принадлежности')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'graph/plot_trapezoidal/{column}_mf.png', dpi=200)


def parabolic_membership_function(x, v, w):
    return max(1 - ((x - v) / w) ** 2, 0)


def define_parabolic_terms(df, column):
    min_value = df[column].min()
    max_value = df[column].max()
    range_value = max_value - min_value
    # Предположим, что ширина параболы w будет половиной от диапазона значений
    w = range_value / 2
    return [
        {'name': 'Очень низкий', 'v': min_value, 'w': w},
        {'name': 'Низкий', 'v': min_value + range_value * 0.25, 'w': w},
        {'name': 'Средний', 'v': min_value + range_value * 0.5, 'w': w},
        {'name': 'Высокий', 'v': min_value + range_value * 0.75, 'w': w},
        {'name': 'Очень высокий', 'v': max_value, 'w': w}
    ]


def plot_parabolic_mfs(df, column, terms):
    x_values = df[column].sort_values().unique()

    plt.figure(figsize=(10, 5))

    for term in terms:
        y_values = [parabolic_membership_function(x, term['v'], term['w']) for x in x_values]
        plt.plot(x_values, y_values, label=term['name'])

    plt.title(f'Параболические функции принадлежности для {column}')
    plt.xlabel('Значение переменной')
    plt.ylabel('Степень принадлежности')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'graph/plot_parabolic/{column}_mf.png', dpi=200)


def gaussian_membership_function(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))


def define_gaussian_terms(df, column):
    min_value = df[column].min()
    max_value = df[column].max()
    range_value = max_value - min_value
    # Определяем стандартное отклонение так, чтобы оно охватывало большую часть диапазона
    sigma = range_value / 6  # Чтобы около 99% значений попадали внутри +/- 3 sigma
    return [
        {'name': 'Очень низкий', 'c': min_value, 'sigma': sigma},
        {'name': 'Низкий', 'c': min_value + range_value * 0.25, 'sigma': sigma},
        {'name': 'Средний', 'c': min_value + range_value * 0.5, 'sigma': sigma},
        {'name': 'Высокий', 'c': min_value + range_value * 0.75, 'sigma': sigma},
        {'name': 'Очень высокий', 'c': max_value, 'sigma': sigma}
    ]


def plot_gaussian_mfs(df, column, terms):
    x_values = df[column].sort_values().unique()

    plt.figure(figsize=(10, 5))

    for term in terms:
        y_values = [gaussian_membership_function(x, term['c'], term['sigma']) for x in x_values]
        plt.plot(x_values, y_values, label=term['name'])

    plt.title(f'Гауссовы функции принадлежности для {column}')
    plt.xlabel('Значение переменной')
    plt.ylabel('Степень принадлежности')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'graph/plot_gaussian/{column}_mf.png', dpi=200)


def create_backup_rules(ws, atmos_pres, dew_point, rh, temp_target):
    terms = ['Очень низкий', 'Низкий', 'Средний', 'Высокий', 'Очень высокий']
    term_values = {term: i for i, term in enumerate(terms)}
    output_terms = len(terms)
    rules = []

    # Генерация всех возможных комбинаций термов для входных переменных
    for ws_term in terms:
        for atmos_pres_term in terms:
            for dew_point_term in terms:
                for rh_term in terms:
                    # Определение уровня выходного терма на основе среднего значения входных термов
                    avg_level = np.mean([term_values[ws_term], term_values[atmos_pres_term],
                                         term_values[dew_point_term], term_values[rh_term]])
                    # Преобразование среднего уровня к ближайшему выходному терму
                    output_term_index = int(round(avg_level * (output_terms - 1) / (len(term_values) - 1)))
                    output_term = terms[output_term_index]

                    # Создание правила
                    rule = ctrl.Rule(ws[ws_term] & atmos_pres[atmos_pres_term] &
                                     dew_point[dew_point_term] & rh[rh_term],
                                     temp_target[output_term])
                    rules.append(rule)

    return rules


def create_fuzzy_system(df_cleaned):
    # Нечеткие входные переменные
    ws = ctrl.Antecedent(np.arange(df_cleaned['ws'].min(), df_cleaned['ws'].max(), 1), 'ws')
    atmos_pres = ctrl.Antecedent(np.arange(df_cleaned['atmos_pres'].min(), df_cleaned['atmos_pres'].max(), 1),
                                 'atmos_pres')
    dew_point = ctrl.Antecedent(np.arange(df_cleaned['dew_point'].min(), df_cleaned['dew_point'].max(), 1), 'dew_point')
    rh = ctrl.Antecedent(np.arange(df_cleaned['rh'].min(), df_cleaned['rh'].max(), 1), 'rh')

    # Нечеткая выходная переменная
    temp_target = ctrl.Consequent(np.arange(df_cleaned['temp'].min(), df_cleaned['temp'].max(), 1), 'temp_target')

    # Определение функций принадлежности на основе ранее определенных термов
    ws_terms = define_triangular_terms(df_cleaned, 'ws')
    atmos_pres_terms = define_triangular_terms(df_cleaned, 'atmos_pres')
    dew_point_terms = define_triangular_terms(df_cleaned, 'dew_point')
    rh_terms = define_triangular_terms(df_cleaned, 'rh')
    temp_target_terms = define_triangular_terms(df_cleaned, 'temp')

    # Добавление функций принадлежности для каждого терма
    for term in ws_terms:
        ws[term['name']] = fuzz.trimf(ws.universe, [term['a'], term['b'], term['c']])
    for term in atmos_pres_terms:
        atmos_pres[term['name']] = fuzz.trimf(atmos_pres.universe, [term['a'], term['b'], term['c']])
    for term in dew_point_terms:
        dew_point[term['name']] = fuzz.trimf(dew_point.universe, [term['a'], term['b'], term['c']])
    for term in rh_terms:
        rh[term['name']] = fuzz.trimf(rh.universe, [term['a'], term['b'], term['c']])
    for term in temp_target_terms:
        temp_target[term['name']] = fuzz.trimf(temp_target.universe, [term['a'], term['b'], term['c']])

    # Основные правила
    rule1 = ctrl.Rule(ws['Очень низкий'] &
                      atmos_pres['Очень высокий'] &
                      dew_point['Очень низкий'] &
                      rh['Очень высокий'],
                      temp_target['Очень низкий'])
    rule2 = ctrl.Rule(ws['Низкий'] &
                      atmos_pres['Высокий'] &
                      dew_point['Низкий'] &
                      rh['Высокий'],
                      temp_target['Низкий'])
    rule3 = ctrl.Rule(ws['Средний'] &
                      atmos_pres['Средний'] &
                      dew_point['Средний'] &
                      rh['Средний'],
                      temp_target['Средний'])
    rule4 = ctrl.Rule(ws['Высокий'] &
                      atmos_pres['Низкий'] &
                      dew_point['Высокий'] &
                      rh['Низкий'],
                      temp_target['Высокий'])
    rule5 = ctrl.Rule(ws['Очень высокий'] &
                      atmos_pres['Очень низкий'] &
                      dew_point['Очень высокий'] &
                      rh['Очень низкий'],
                      temp_target['Очень высокий'])

    # Создание резервных правил
    backup_rules = create_backup_rules(ws, atmos_pres, dew_point, rh, temp_target)

    # Создание и запуск системы вывода
    temp_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5] + backup_rules)
    temp_simulation = ctrl.ControlSystemSimulation(temp_ctrl)

    # Применение системы вывода к новым данным
    def predict_temp(ws_val, atmos_pres_val, dew_point_val, rh_val):
        temp_simulation.input['ws'] = ws_val
        temp_simulation.input['atmos_pres'] = atmos_pres_val
        temp_simulation.input['dew_point'] = dew_point_val
        temp_simulation.input['rh'] = rh_val
        temp_simulation.compute()
        return temp_simulation.output['temp_target']

    return predict_temp


if __name__ == "__main__":
    # Путь к файлу
    file_path = 'Dataset_FUERTEVENTURA.csv'

    # Анализ датасета
    analyze_dataset(file_path)

    # Загрузка датасета
    df = pd.read_csv(file_path)

    # Определение столбцов для анализа
    columns_for_analysis = [
        'temp',
        'wd',
        'ws',
        'atmos_pres',
        'dew_point',
        'rh',
        'ceil_hgt',
        'visibility',
    ]

    # Удаление строк с пропущенными значениями в интересующих столбцах
    df_cleaned = delete_rows_with_missing_values(df, columns_for_analysis)

    # Отображение тепловой карты корреляции
    display_heatmap(df_cleaned, columns_for_analysis)

    for column in columns_for_analysis:
        terms = define_triangular_terms(df, column)
        plot_triangular_mfs(df, column, terms)

    for column in columns_for_analysis:
        terms = define_trapezoidal_terms(df, column)
        plot_trapezoidal_mfs(df, column, terms)

    for column in columns_for_analysis:
        terms = define_parabolic_terms(df_cleaned, column)
        plot_parabolic_mfs(df_cleaned, column, terms)

    for column in columns_for_analysis:
        terms = define_gaussian_terms(df_cleaned, column)
        plot_gaussian_mfs(df_cleaned, column, terms)

    predict_temp = create_fuzzy_system(df_cleaned)
    print("Прогнозируемая температура при ws=5, atmos_pres=1010, dew_point=10, rh=60:", predict_temp(3, 1070, 0, 90))

    predictions = []
    real_values = df_cleaned['temp']
    for index, row in df_cleaned.iterrows():
        prediction = predict_temp(row['ws'], row['atmos_pres'], row['dew_point'], row['rh'])
        predictions.append(prediction)

    # Оценка модели
    mse = mean_squared_error(real_values, predictions)
    r2 = r2_score(real_values, predictions)

    print(mse)
    print(r2)
