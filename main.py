import pandas as pd
import numpy as np
from datetime import datetime

print("=== ВЫПОЛНЕНИЕ ВСЕХ 4 ЗАДАНИЙ ===\n")

# =============================================================================
# ЗАДАНИЕ 1: "Первичный анализ и очистка данных о сотрудниках"
# =============================================================================
print("ЗАДАНИЕ 1: ПЕРВИЧНЫЙ АНАЛИЗ И ОЧИСТКА ДАННЫХ О СОТРУДНИКАХ")

try:

    df_employees = pd.read_csv('employees_dataset.csv')
    print("1. Структура данных:")
    print(f"   Размер: {df_employees.shape}")
    print(f"   Столбцы: {list(df_employees.columns)}")
    print(df_employees.head())


    print("\n2. Анализ проблем:")
    print("   Пропущенные значения:")
    print(df_employees.isnull().sum())


    print(f"   Тип данных Age: {df_employees['Age'].dtype}")


    duplicates = df_employees.duplicated().sum()
    print(f"   Явные дубликаты: {duplicates}")


    df_employees.columns = [col.lower().replace(' ', '_') for col in df_employees.columns]
    print("\n3. Столбцы приведены к snake_case")


    df_employees = df_employees.drop_duplicates()
    print(f"4. Удалено дубликатов: {duplicates}")


    df_employees['age'] = pd.to_numeric(df_employees['age'], errors='coerce')
    print("5. Столбец age преобразован в числовой тип")
    print(f"   Нечисловые значения заменены на NaN: {df_employees['age'].isna().sum()}")


    df_employees['city'] = df_employees['city'].fillna('Не указан')
    print("6. Пропуски в city заполнены значением 'Не указан'")

    print("\n✓ ЗАДАНИЕ 1 ВЫПОЛНЕНО")
    print(f"Итоговый размер: {df_employees.shape}")

except FileNotFoundError:
    print("❌ Файл employees_dataset.csv не найден")

# =============================================================================
# ЗАДАНИЕ 2: "Обработка и анализ продаж интернет-магазина"
# =============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: ОБРАБОТКА И АНАЛИЗ ПРОДАЖ ИНТЕРНЕТ-МАГАЗИНА")

try:

    df_sales = pd.read_csv('sales_dataset.csv')
    print(f"1. Загружено записей: {len(df_sales)}")


    city_mapping = {
        'NY': 'New York', 'NYC': 'New York', 'new york': 'New York',
        'LDN': 'London', 'london': 'London',
        'BER': 'Berlin', 'berlin': 'Berlin'
    }
    df_sales['customer_city'] = df_sales['customer_city'].replace(city_mapping)
    print("2. Названия городов стандартизированы")


    df_sales['customer_city'] = df_sales['customer_city'].fillna('Unknown')
    discount_mean = df_sales['discount'].mean()
    df_sales['discount'] = df_sales['discount'].fillna(discount_mean)
    print(f"3. Пропуски заполнены (средняя скидка: {discount_mean:.2f})")


    conditions = [
        df_sales['discount'] == 0,
        (df_sales['discount'] > 0) & (df_sales['discount'] <= 0.1),
        df_sales['discount'] > 0.1
    ]
    choices = ['Без скидки', 'Маленькая', 'Большая']
    df_sales['discount_group'] = np.select(conditions, choices, default='Неизвестно')
    print("4. Создан столбец discount_group")


    df_sales['revenue'] = df_sales['quantity'] * df_sales['price'] * (1 - df_sales['discount'])
    revenue_by_product = df_sales.groupby('product')['revenue'].sum().sort_values(ascending=False)


    avg_receipt_by_city = df_sales.groupby('customer_city')['revenue'].mean().sort_values(ascending=False)

    print("\n5. РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("   Общая выручка по продуктам:")
    print(revenue_by_product.head())
    print(
        f"\n   Город с самым высоким средним чеком: {avg_receipt_by_city.index[0]} ({avg_receipt_by_city.iloc[0]:.2f})")

    print("\n✓ ЗАДАНИЕ 2 ВЫПОЛНЕНО")

except FileNotFoundError:
    print("❌ Файл sales_dataset.csv не найден")

# =============================================================================
# ЗАДАНИЕ 3: "Статистический анализ результатов экспериментов"
# =============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 3: СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")

try:

    df_experiments = pd.read_csv('experiments_dataset.csv')
    print(f"1. Загружено экспериментов: {len(df_experiments)}")


    print("\n2. ОБРАБОТКА ПРОПУСКОВ:")
    print("   Пропуски до обработки:")
    print(df_experiments.isnull().sum())

    temp_mean = df_experiments['temperature'].mean()
    success_median = df_experiments['success_rate'].median()

    df_experiments['temperature'] = df_experiments['temperature'].fillna(temp_mean)
    df_experiments['success_rate'] = df_experiments['success_rate'].fillna(success_median)

    print(f"   Заполнено temperature средним: {temp_mean:.2f}")
    print(f"   Заполнено success_rate медианой: {success_median:.2f}")


    print("\n3. СТАТИСТИКА ПО ЭКСПЕРИМЕНТАЛЬНЫМ ГРУППАМ:")

    group_stats = {}
    for group in df_experiments['research_group'].unique():
        group_data = df_experiments[df_experiments['research_group'] == group]

        mean_temp = np.mean(group_data['temperature'])
        std_success = np.std(group_data['success_rate'])
        total_experiments = len(group_data)

        group_stats[group] = {
            'mean_temperature': mean_temp,
            'std_success_rate': std_success,
            'total_experiments': total_experiments
        }

        print(f"   Группа {group}:")
        print(f"     Средняя температура: {mean_temp:.2f}")
        print(f"     Стандартное отклонение успешности: {std_success:.2f}")
        print(f"     Всего экспериментов: {total_experiments}")


    measurement_cols = [f'measurement_{i}' for i in range(1, 6)]
    df_experiments['avg_measurement'] = df_experiments[measurement_cols].mean(axis=1)

    researcher_avg = df_experiments.groupby('researcher')['avg_measurement'].mean().sort_values(ascending=False)
    best_researcher = researcher_avg.index[0]
    best_avg = researcher_avg.iloc[0]

    print(f"\n4. ЛУЧШИЙ ИССЛЕДОВАТЕЛЬ: {best_researcher} (средний показатель: {best_avg:.2f})")

    print("\n✓ ЗАДАНИЕ 3 ВЫПОЛНЕНО")

except FileNotFoundError:
    print("❌ Файл experiments_dataset.csv не найден")

# =============================================================================
# ЗАДАНИЕ 4: "Комплексная подготовка медицинских данных для анализа"
# =============================================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 4: КОМПЛЕКСНАЯ ПОДГОТОВКА МЕДИЦИНСКИХ ДАННЫХ")

try:

    df_medical = pd.read_csv('medical_dataset.csv')
    print(f"1. Загружено медицинских записей: {len(df_medical)}")


    df_medical.columns = [col.lower().replace(' ', '_') for col in df_medical.columns]
    print("2. Заголовки приведены к snake_case")


    df_medical['medication'] = df_medical['medication'].fillna('Not Prescribed')


    df_medical['heart_rate_temp'] = pd.to_numeric(df_medical['heart_rate'], errors='coerce')
    heart_rate_median = df_medical['heart_rate_temp'].median()
    df_medical['heart_rate'] = df_medical['heart_rate_temp'].fillna(heart_rate_median)
    df_medical = df_medical.drop('heart_rate_temp', axis=1)

    print(f"3. Пропуски обработаны (медиана heart_rate: {heart_rate_median:.1f})")


    initial_count = len(df_medical)
    df_medical = df_medical.drop_duplicates()
    print(f"4. Удалено дубликатов: {initial_count - len(df_medical)}")


    condition_mapping = {
        'High BP': 'Hypertension',
        'HTN': 'Hypertension',
        'Sugar': 'Diabetes',
        'DM': 'Diabetes',
        'Breathing issues': 'Asthma'
    }
    df_medical['condition'] = df_medical['condition'].replace(condition_mapping)
    print("5. Условия стандартизированы")


    df_medical['date_of_birth'] = pd.to_datetime(df_medical['date_of_birth'], errors='coerce')
    df_medical['visit_date'] = pd.to_datetime(df_medical['visit_date'], errors='coerce', format='mixed')
    print("6. Даты преобразованы в datetime")


    print("7. Heart_rate преобразован в числовой тип")


    current_date = pd.Timestamp.now()
    df_medical['age'] = (
                (df_medical['visit_date'].fillna(current_date) - df_medical['date_of_birth']).dt.days / 365.25).round(1)
    print(f"8. Создан столбец age (диапазон: {df_medical['age'].min():.1f}-{df_medical['age'].max():.1f} лет)")


    doctor_stats = df_medical.groupby('doctor').agg({
        'age': 'mean',
        'visit_date': 'count',
        'patientid': 'nunique'
    }).round(2)
    doctor_stats.columns = ['average_age', 'total_visits', 'unique_patients']

    print("\n9. СТАТИСТИКА ПО ВРАЧАМ:")
    print(doctor_stats)

    med_a_analysis = df_medical[df_medical['medication'] == 'Med_A'].groupby('condition').size().sort_values(
        ascending=False)

    if not med_a_analysis.empty:
        most_common_condition = med_a_analysis.index[0]
        med_a_count = med_a_analysis.iloc[0]
        print(f"\n10. Med_A чаще всего назначается при: {most_common_condition} ({med_a_count} назначений)")
    else:
        print("\n10. Назначения Med_A не найдены")

    print("\n✓ ЗАДАНИЕ 4 ВЫПОЛНЕНО")

except FileNotFoundError:
    print("❌ Файл medical_dataset.csv не найден")

# =============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n" + "=" * 60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")

try:
    df_employees.to_csv('employees_cleaned.csv', index=False)
    print("✓ employees_cleaned.csv - очищенные данные сотрудников")
except:
    pass

try:
    df_sales.to_csv('sales_analyzed.csv', index=False)
    print("✓ sales_analyzed.csv - проанализированные данные продаж")
except:
    pass

try:
    df_experiments.to_csv('experiments_analyzed.csv', index=False)
    print("✓ experiments_analyzed.csv - проанализированные эксперименты")
except:
    pass

try:
    df_medical.to_csv('medical_processed.csv', index=False)
    print("✓ medical_processed.csv - обработанные медицинские данные")
except:
    pass

print("\n=== ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ ===")