import os
import json
import gdown
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

for file in os.listdir():
    if "data"  == file:
        break
else:
    os.makedirs("data")

files_id = ["1ERBzQWURAhvxePxMWaBCmirR7NwORt8F"]
file_path = os.path.join("data", "food_delivery_analytics_cleaned.csv")
for file_id in files_id:
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        file_path,
        quiet=False
    )

try:
    df = pd.read_csv(file_path)
    print('Arquivo carregado com sucesso!')
    print(df.head())
except FileNotFoundError:
    print(f"Erro: arquivo não encontrado em: {file_path}")
except Exception as e:
    print(f"Erro ao ler o arquivo: {e}")

df.info()

print(df.isnull().sum() / len(df) *100 )

print(df.shape)

print(df.describe())

bool_cols = ['cancellation_flag','delayed_delivery_flag','refund_flag','promo_code_used','premium_customer_flag','festival_or_weekend_flag']

print('Proporcao entre classes (variaveis booleanas):')
for col in bool_cols:
    counts = df[col].value_counts()
    pct = df[col].value_counts(normalize=True) * 100
    print(f'{col}:')
    for val in counts.index:
        print(f'  {val}: {counts[val]} ({pct[val]:.1f}%)')
    print()


bins = int(np.ceil(np.log2(len(df['customer_age'])) + 1))

print(df['customer_age'].describe())
print(f"Assimetria (Skewness): {df['customer_age'].skew():.4f} | Curtose (Kurtosis): {df['customer_age'].kurt():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['customer_age'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['customer_age'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['customer_age'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['customer_loyalty_score'].describe())
print(f"Assimetria (Skewness): {df['customer_loyalty_score'].skew():.4f} | Curtose (Kurtosis): {df['customer_loyalty_score'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['customer_loyalty_score'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['customer_loyalty_score'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['customer_loyalty_score'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['customer_loyalty_score'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['order_hour'].describe())
print(f"Assimetria (Skewness): {df['order_hour'].skew():.4f} | Curtose (Kurtosis): {df['order_hour'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['order_hour'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['order_hour'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['order_hour'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['order_hour'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['delivery_distance_km'].describe())
print(f"Assimetria (Skewness): {df['delivery_distance_km'].skew():.4f} | Curtose (Kurtosis): {df['delivery_distance_km'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['delivery_distance_km'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['delivery_distance_km'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['delivery_distance_km'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['delivery_distance_km'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['delivery_time_minutes'].describe())
print(f"Assimetria (Skewness): {df['delivery_time_minutes'].skew():.4f} | Curtose (Kurtosis): {df['delivery_time_minutes'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['delivery_time_minutes'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['delivery_time_minutes'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['delivery_time_minutes'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['delivery_time_minutes'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['preparation_time_minutes'].describe())
print(f"Assimetria (Skewness): {df['preparation_time_minutes'].skew():.4f} | Curtose (Kurtosis): {df['preparation_time_minutes'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['preparation_time_minutes'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['preparation_time_minutes'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['preparation_time_minutes'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['preparation_time_minutes'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['traffic_level_score'].describe())
print(f"Assimetria (Skewness): {df['traffic_level_score'].skew():.4f} | Curtose (Kurtosis): {df['traffic_level_score'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['traffic_level_score'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['traffic_level_score'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['traffic_level_score'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['traffic_level_score'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['weather_severity_score'].describe())
print(f"Assimetria (Skewness): {df['weather_severity_score'].skew():.4f} | Curtose (Kurtosis): {df['weather_severity_score'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['weather_severity_score'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['weather_severity_score'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['weather_severity_score'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['weather_severity_score'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['restaurant_rating'].describe())
print(f"Assimetria (Skewness): {df['restaurant_rating'].skew():.4f} | Curtose (Kurtosis): {df['restaurant_rating'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['restaurant_rating'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['restaurant_rating'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['restaurant_rating'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['restaurant_rating'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['customer_rating'].describe())
print(f"Assimetria (Skewness): {df['customer_rating'].skew():.4f} | Curtose (Kurtosis): {df['customer_rating'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['customer_rating'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['customer_rating'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['customer_rating'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['customer_rating'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['order_value'].describe())
print(f"Assimetria (Skewness): {df['order_value'].skew():.4f} | Curtose (Kurtosis): {df['order_value'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['order_value'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['order_value'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['order_value'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['order_value'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['delivery_fee'].describe())
print(f"Assimetria (Skewness): {df['delivery_fee'].skew():.4f} | Curtose (Kurtosis): {df['delivery_fee'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['delivery_fee'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['delivery_fee'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['delivery_fee'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['delivery_fee'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['discount_amount'].describe())
print(f"Assimetria (Skewness): {df['discount_amount'].skew():.4f} | Curtose (Kurtosis): {df['discount_amount'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['discount_amount'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['discount_amount'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['discount_amount'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['discount_amount'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['number_of_items'].describe())
print(f"Assimetria (Skewness): {df['number_of_items'].skew():.4f} | Curtose (Kurtosis): {df['number_of_items'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['number_of_items'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['number_of_items'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['number_of_items'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['number_of_items'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['delayed_delivery_flag'].unique())
print(df['delayed_delivery_flag'].value_counts())
print(df['delayed_delivery_flag'].value_counts() / len(df)*100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(
    x='delayed_delivery_flag',
    data=df,
    ax=axes[0]
)

for p in axes[0].patches:
    axes[0].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom'
    )

axes[0].set_title('Distribuição de Entregas atrasadas')
axes[0].set_xlabel('Atraso na Entrega')
axes[0].set_ylabel('Quantidade')
axes[0].patches[0].set_facecolor('steelblue')   
axes[0].patches[1].set_facecolor('orange')

df['delayed_delivery_flag'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=axes[1]
)

axes[1].set_title('Proporção de Entregas Atrasadas')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print(df['promo_code_used'].unique())
print(df['promo_code_used'].value_counts())
print(df['promo_code_used'].value_counts() / len(df)*100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(
    x='promo_code_used',
    data=df,
    ax=axes[0]
)

for p in axes[0].patches:
    axes[0].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom'
    )

axes[0].set_title('Distribuição de Uso de Código Promocional')
axes[0].set_xlabel('Código Promocional')
axes[0].set_ylabel('Quantidade')
axes[0].patches[0].set_facecolor('steelblue')  
axes[0].patches[1].set_facecolor('orange') 

df['promo_code_used'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=axes[1]
)

axes[1].set_title('Proporção de Uso de Código Promocional')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print(df['tip_amount'].describe())
print(f"Assimetria (Skewness): {df['tip_amount'].skew():.4f} | Curtose (Kurtosis): {df['tip_amount'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['tip_amount'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['tip_amount'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['tip_amount'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['tip_amount'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['final_amount_paid'].describe())
print(f"Assimetria (Skewness): {df['final_amount_paid'].skew():.4f} | Curtose (Kurtosis): {df['final_amount_paid'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['final_amount_paid'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['final_amount_paid'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['final_amount_paid'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['final_amount_paid'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

print(df['order_day_of_week'].unique())
print(df['order_day_of_week'].value_counts())
print(df['order_day_of_week'].value_counts() / len(df)*100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(
    x='order_day_of_week',
    data=df,
    ax=axes[0]
)

for p in axes[0].patches:
    axes[0].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom'
    )

axes[0].set_title('Distribuição Compras no dia da semana')
axes[0].set_xlabel('Dia da Semana')
axes[0].set_ylabel('Quantidade')
axes[0].patches[0].set_facecolor('steelblue')
axes[0].patches[1].set_facecolor('orange') 

df['order_day_of_week'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=axes[1]
)

axes[1].set_title('Proporção de Compras no dia da semana')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print(df['delivery_partner_rating'].describe())
print(f"Assimetria (Skewness): {df['delivery_partner_rating'].skew():.4f} | Curtose (Kurtosis): {df['delivery_partner_rating'].kurt():.4f}")
bins = int(np.ceil(np.log2(len(df['delivery_partner_rating'])) + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['delivery_partner_rating'], bins=bins, ax=axes[0])
axes[0].set_title('Histograma')
sns.kdeplot(df['delivery_partner_rating'], fill=True, ax=axes[1])
axes[1].set_title('Densidade (KDE)')
sns.boxplot(x=df['delivery_partner_rating'], ax=axes[2])
axes[2].set_title('Boxplot')
plt.tight_layout()
plt.show()

num_cols = [
    'customer_age',
    'customer_loyalty_score',
    'order_hour',
    'delivery_distance_km',
    'delivery_time_minutes',
    'estimated_delivery_time',
    'traffic_level_score',
    'weather_severity_score',
    'restaurant_rating',
    'delivery_partner_rating',
    'customer_rating',
    'order_value',
    'delivery_fee',
    'discount_amount',
    'tip_amount',
    'final_amount_paid',
    'number_of_items',
    'delivery_partner_rating'
]

correlation_matrix = df[num_cols].corr()
print(correlation_matrix)

def boolean_numeric_analysis(df, bool_col, num_col):
    summary = df.groupby(bool_col)[num_col].agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max'
    ).reset_index()

    return summary

boolean_numeric_analysis(df, 'delayed_delivery_flag', 'customer_rating')

data = df[['delivery_partner_rating', 'delayed_delivery_flag']].dropna()
data['delayed_delivery_flag'] = data['delayed_delivery_flag'].astype(bool)

not_delayed = data[data['delayed_delivery_flag'] == False]['delivery_partner_rating']
delayed = data[data['delayed_delivery_flag'] == True]['delivery_partner_rating']

summary = data.groupby('delayed_delivery_flag')['delivery_partner_rating'].agg(
    count='count',
    mean='mean',
    median='median',
    std='std',
    min='min',
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75),
    max='max'
).reset_index()

print(summary)

print(f'Média sem atraso: {not_delayed.mean():.4f}')
print(f'Média com atraso: {delayed.mean():.4f}')
print(f'Diferença de médias: {delayed.mean() - not_delayed.mean():.4f}')

print(f'Mediana sem atraso: {not_delayed.median():.4f}')
print(f'Mediana com atraso: {delayed.median():.4f}')
print(f'Diferença de medianas: {delayed.median() - not_delayed.median():.4f}')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc = axes[0].scatter(
    df['delivery_distance_km'], df['delivery_time_minutes'],
    c=df['traffic_level_score'], cmap='RdYlGn_r', alpha=0.4, s=10
)
plt.colorbar(sc, ax=axes[0], label='Nível de Tráfego')
corr = df['delivery_distance_km'].corr(df['delivery_time_minutes'])
axes[0].annotate(f'Correlação: {corr:.4f}', xy=(0.05, 0.92), xycoords='axes fraction')
axes[0].set_xlabel('Distância de Entrega (km)')
axes[0].set_ylabel('Tempo de Entrega (min)')
axes[0].set_title('Distância vs Tempo por Nível de Tráfego')

df['traffic_group'] = pd.cut(df['traffic_level_score'], bins=[0, 3.3, 6.6, 10],
                              labels=['Baixo (0–3.3)', 'Médio (3.3–6.6)', 'Alto (6.6–10)'])
sns.boxplot(data=df, x='traffic_group', y='delivery_time_minutes', ax=axes[1])
axes[1].set_xlabel('Faixa de Tráfego')
axes[1].set_ylabel('Tempo de Entrega (min)')
axes[1].set_title('Tempo de Entrega por Faixa de Tráfego')

plt.tight_layout()
plt.show()

print(df.groupby('traffic_group', observed=True)['delivery_time_minutes'].agg(['mean', 'median']))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x='restaurant_rating', y='customer_rating', alpha=0.3, ax=axes[0])
corr_val = df['restaurant_rating'].corr(df['customer_rating'])
axes[0].annotate(f'Correlação: {corr_val:.4f}', xy=(0.05, 0.92), xycoords='axes fraction')
axes[0].set_xlabel('Nota do Restaurante')
axes[0].set_ylabel('Nota do Cliente')
axes[0].set_title('Nota do Restaurante vs Nota do Cliente')

sns.regplot(data=df, x='restaurant_rating', y='customer_rating',
            scatter_kws={'alpha': 0.2, 's': 10}, line_kws={'color': 'red'}, ax=axes[1])
axes[1].set_xlabel('Nota do Restaurante')
axes[1].set_ylabel('Nota do Cliente')
axes[1].set_title('Regressão: Nota do Restaurante vs Nota do Cliente')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df, x='promo_code_used', y='order_value', ax=axes[0])
axes[0].set_xticklabels(['Sem código', 'Com código'])
axes[0].set_xlabel('Uso de Código Promocional')
axes[0].set_ylabel('Valor do Pedido (R$)')
axes[0].set_title('Valor do Pedido por Uso de Código Promocional')

print(df.groupby('promo_code_used')['order_value'].agg(['mean', 'median', 'std']))

sns.violinplot(data=df, x='promo_code_used', y='order_value', inner='quartile', ax=axes[1])
axes[1].set_xticklabels(['Sem código', 'Com código'])
axes[1].set_xlabel('Uso de Código Promocional')
axes[1].set_ylabel('Valor do Pedido (R$)')
axes[1].set_title('Distribuição do Valor do Pedido por Código Promocional')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x='number_of_items', y='final_amount_paid', alpha=0.3, ax=axes[0])
corr_val = df['number_of_items'].corr(df['final_amount_paid'])
axes[0].annotate(f'Correlação: {corr_val:.4f}', xy=(0.05, 0.92), xycoords='axes fraction')
axes[0].set_xlabel('Número de Itens')
axes[0].set_ylabel('Valor Final Pago (R$)')
axes[0].set_title('Quantidade de Itens vs Valor Final Pago')

mean_by_items = df.groupby('number_of_items')['final_amount_paid'].mean()
axes[1].bar(mean_by_items.index, mean_by_items.values, color='steelblue', edgecolor='white')
axes[1].set_xlabel('Número de Itens')
axes[1].set_ylabel('Valor Médio Final Pago (R$)')
axes[1].set_title('Valor Médio por Quantidade de Itens')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(18, 10))

sns.boxplot(data=df, x='order_hour', y='delivery_time_minutes', ax=axes[0])
axes[0].set_xlabel('Hora do Pedido')
axes[0].set_ylabel('Tempo de Entrega (min)')
axes[0].set_title('Tempo de Entrega por Hora do Pedido')

mean_by_hour = df.groupby('order_hour')['delivery_time_minutes'].mean()
axes[1].plot(mean_by_hour.index, mean_by_hour.values, marker='o', color='steelblue')
axes[1].axhline(df['delivery_time_minutes'].mean(), color='red', linestyle='--', label='Média geral')
axes[1].set_xlabel('Hora do Pedido')
axes[1].set_ylabel('Tempo Médio de Entrega (min)')
axes[1].set_title('Tempo Médio de Entrega por Hora')
axes[1].set_xticks(range(0, 24))
axes[1].legend()

plt.tight_layout()
plt.show()

corr_val = df['order_hour'].corr(df['delivery_time_minutes'])
print(f"Correlação hora do pedido vs tempo de entrega: {corr_val:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x='weather_severity_score', y='delivery_time_minutes', alpha=0.3, ax=axes[0])
corr_val = df['weather_severity_score'].corr(df['delivery_time_minutes'])
axes[0].annotate(f'Correlação: {corr_val:.4f}', xy=(0.05, 0.92), xycoords='axes fraction')
axes[0].set_xlabel('Severidade Climática')
axes[0].set_ylabel('Tempo de Entrega (min)')
axes[0].set_title('Severidade Climática vs Tempo de Entrega')

sns.regplot(data=df, x='weather_severity_score', y='delivery_time_minutes',
            scatter_kws={'alpha': 0.2, 's': 10}, line_kws={'color': 'red'}, ax=axes[1])
axes[1].set_xlabel('Severidade Climática')
axes[1].set_ylabel('Tempo de Entrega (min)')
axes[1].set_title('Regressão: Severidade Climática vs Tempo de Entrega')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x='customer_loyalty_score', y='customer_rating', alpha=0.3, ax=axes[0])
corr_val = df['customer_loyalty_score'].corr(df['customer_rating'])
axes[0].annotate(f'Correlação: {corr_val:.4f}', xy=(0.05, 0.92), xycoords='axes fraction')
axes[0].set_xlabel('Pontuação de Lealdade')
axes[0].set_ylabel('Nota do Cliente')
axes[0].set_title('Lealdade vs Nota do Cliente')

df['loyalty_quartile'] = pd.qcut(df['customer_loyalty_score'], q=4,
                                  labels=['Q1 (Baixa)', 'Q2', 'Q3', 'Q4 (Alta)'])
sns.boxplot(data=df, x='loyalty_quartile', y='customer_rating', ax=axes[1])
axes[1].set_xlabel('Quartil de Lealdade')
axes[1].set_ylabel('Nota do Cliente')
axes[1].set_title('Nota do Cliente por Quartil de Lealdade')

plt.tight_layout()
plt.show()

print(df.groupby('loyalty_quartile', observed=True)['customer_rating'].agg(['mean', 'median', 'std']))

num_cols = [
    'customer_age',
    'customer_loyalty_score',
    'order_hour',
    'delivery_distance_km',
    'delivery_time_minutes',
    'estimated_delivery_time',
    'traffic_level_score',
    'weather_severity_score',
    'restaurant_rating',
    'delivery_partner_rating',
    'customer_rating',
    'order_value',
    'delivery_fee',
    'discount_amount',
    'tip_amount',
    'final_amount_paid',
    'number_of_items'
]

corr = df[num_cols].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Matriz de correlação entre variáveis numéricas')
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))

sns.violinplot(
    data=data,
    x='delayed_delivery_flag',
    y='delivery_partner_rating',
    inner='quartile',
    palette={'False': 'steelblue', 'True': 'orange'},
    ax=ax
)

ax.set_xticklabels(['Sem atraso', 'Com atraso'])
ax.set_xlabel('Status da Entrega', fontsize=13)
ax.set_ylabel('Avaliação do Parceiro de Entrega', fontsize=13)
ax.set_title('Distribuição da Avaliação do Entregador por Status de Atraso',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 7))

sc = ax.scatter(
    df['delivery_distance_km'], df['delivery_time_minutes'],
    c=df['traffic_level_score'], cmap='RdYlGn_r',
    alpha=0.5, s=12, edgecolors='none'
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Nível de Tráfego (1 = baixo, 10 = alto)', fontsize=11)

ax.set_xlabel('Distância de Entrega (km)', fontsize=13)
ax.set_ylabel('Tempo de Entrega (min)', fontsize=13)
ax.set_title('Impacto da Distância e do Tráfego no Tempo de Entrega',
             fontsize=15, fontweight='bold')

corr = df['delivery_distance_km'].corr(df['delivery_time_minutes'])
ax.annotate(
    f'Correlação distância–tempo: {corr:.2f}',
    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=11,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
)

plt.tight_layout()
plt.show()

mean_by_hour = df.groupby('order_hour')['delivery_time_minutes'].mean().reset_index()
overall_mean = df['delivery_time_minutes'].mean()

fig, ax = plt.subplots(figsize=(14, 6))

colors = ['#e74c3c' if v > overall_mean else '#2ecc71'
          for v in mean_by_hour['delivery_time_minutes']]
ax.bar(mean_by_hour['order_hour'], mean_by_hour['delivery_time_minutes'],
       color=colors, edgecolor='white', linewidth=0.5)

ax.axhline(overall_mean, color='navy', linestyle='--', linewidth=1.5)
ax.set_xlabel('Hora do Pedido', fontsize=13)
ax.set_ylabel('Tempo Médio de Entrega (min)', fontsize=13)
ax.set_title('Tempo Médio de Entrega por Hora do Pedido', fontsize=15, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f'{h:02d}h' for h in range(24)], rotation=45, ha='right')
ax.set_ylim(mean_by_hour['delivery_time_minutes'].min() * 0.9,
            mean_by_hour['delivery_time_minutes'].max() * 1.05)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Acima da média'),
    Patch(facecolor='#2ecc71', label='Abaixo da média'),
    plt.Line2D([0], [0], color='navy', linestyle='--', label=f'Média geral: {overall_mean:.1f} min')
]
ax.legend(handles=legend_elements, fontsize=11)

plt.tight_layout()
plt.show()

df['weather_group'] = pd.cut(
    df['weather_severity_score'],
    bins=[0, 2.5, 5.0, 7.5, 10.0],
    labels=['Leve (0–2.5)', 'Moderado (2.5–5)', 'Intenso (5–7.5)', 'Severo (7.5–10)'],
    include_lowest=True
)

palette = {
    'Leve (0–2.5)': '#2ecc71', 'Moderado (2.5–5)': '#f1c40f',
    'Intenso (5–7.5)': '#e67e22', 'Severo (7.5–10)': '#e74c3c'
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df, x='weather_group', y='delivery_time_minutes', palette=palette, ax=axes[0])
axes[0].set_xlabel('Condição Climática', fontsize=12)
axes[0].set_ylabel('Tempo de Entrega (min)', fontsize=12)
axes[0].set_title('Distribuição do Tempo de Entrega por Condição Climática', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=10)

mean_weather = df.groupby('weather_group', observed=True)['delivery_time_minutes'].mean().reset_index()
bar_colors = [palette[g] for g in mean_weather['weather_group']]
axes[1].bar(range(len(mean_weather)), mean_weather['delivery_time_minutes'],
            color=bar_colors, edgecolor='white')
axes[1].set_xticks(range(len(mean_weather)))
axes[1].set_xticklabels(mean_weather['weather_group'], rotation=10)
axes[1].set_xlabel('Condição Climática', fontsize=12)
axes[1].set_ylabel('Tempo Médio de Entrega (min)', fontsize=12)
axes[1].set_title('Tempo Médio por Condição Climática', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, mean_weather['delivery_time_minutes'].max() * 1.15)

for i, row in mean_weather.iterrows():
    axes[1].text(i, row['delivery_time_minutes'] + 0.5,
                 f"{row['delivery_time_minutes']:.1f} min",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

numeric_cols = [
    'customer_age', 'customer_loyalty_score', 'order_hour', 'order_day_of_week',
    'order_month', 'delivery_distance_km', 'preparation_time_minutes',
    'delivery_time_minutes', 'estimated_delivery_time', 'traffic_level_score',
    'weather_severity_score', 'restaurant_rating', 'delivery_partner_rating',
    'customer_rating', 'order_value', 'delivery_fee', 'discount_amount',
    'tip_amount', 'final_amount_paid', 'number_of_items',
    'delivery_partner_experience_years', 'delivery_efficiency_score'
]

boolean_cols = [
    'cancellation_flag', 'delayed_delivery_flag', 'refund_flag',
    'promo_code_used', 'premium_customer_flag', 'festival_or_weekend_flag'
]


numeric_pipeline = Pipeline([
    ('imputer',       SimpleImputer(strategy='median')),
    ('padronizacao',  StandardScaler()),
    ('re_escala',     MinMaxScaler()),
    ('normalizacao',  Normalizer())
])

boolean_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_cols),
    ('boolean', boolean_pipeline, boolean_cols)
])

df_features = df.drop(columns=['order_id']).copy()
df_features[boolean_cols] = df_features[boolean_cols].astype(int)

df_processed_array = preprocessor.fit_transform(df_features)

output_cols = numeric_cols + boolean_cols
df_processed = pd.DataFrame(df_processed_array, columns=output_cols)

print(f"Shape após pipeline: {df_processed.shape}")
print(f"Valores nulos após pipeline: {df_processed.isnull().sum().sum()}")
print("\nEstatísticas após pré-processamento (colunas numéricas):")
print(df_processed[numeric_cols[:5]].describe().round(4))