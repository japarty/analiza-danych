

```python
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv('PUBG_Player_Statistics.csv')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 87898 entries, 0 to 87897
    Columns: 152 entries, player_name to squad_DBNOs
    dtypes: float64(83), int64(68), object(1)
    memory usage: 101.6+ MB
    


```python
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>tracker_id</th>
      <th>solo_KillDeathRatio</th>
      <th>solo_WinRatio</th>
      <th>solo_TimeSurvived</th>
      <th>solo_RoundsPlayed</th>
      <th>solo_Wins</th>
      <th>solo_WinTop10Ratio</th>
      <th>solo_Top10s</th>
      <th>solo_Top10Ratio</th>
      <th>...</th>
      <th>squad_RideDistance</th>
      <th>squad_MoveDistance</th>
      <th>squad_AvgWalkDistance</th>
      <th>squad_AvgRideDistance</th>
      <th>squad_LongestKill</th>
      <th>squad_Heals</th>
      <th>squad_Revives</th>
      <th>squad_Boosts</th>
      <th>squad_DamageDealt</th>
      <th>squad_DBNOs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BreakNeck</td>
      <td>4405</td>
      <td>3.14</td>
      <td>17.65</td>
      <td>18469.14</td>
      <td>17</td>
      <td>3</td>
      <td>0.83</td>
      <td>4</td>
      <td>23.5</td>
      <td>...</td>
      <td>3751590.99</td>
      <td>5194786.58</td>
      <td>2626.97</td>
      <td>4372.64</td>
      <td>536.98</td>
      <td>2186</td>
      <td>234</td>
      <td>1884</td>
      <td>242132.73</td>
      <td>1448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blackwalk</td>
      <td>8199</td>
      <td>4.41</td>
      <td>18.18</td>
      <td>33014.86</td>
      <td>33</td>
      <td>6</td>
      <td>0.36</td>
      <td>11</td>
      <td>33.3</td>
      <td>...</td>
      <td>4295917.30</td>
      <td>6051783.67</td>
      <td>2422.48</td>
      <td>6009.73</td>
      <td>734.85</td>
      <td>2041</td>
      <td>276</td>
      <td>2340</td>
      <td>269795.75</td>
      <td>1724</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mercedes_benz</td>
      <td>4454</td>
      <td>3.60</td>
      <td>0.00</td>
      <td>4330.44</td>
      <td>5</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>20.0</td>
      <td>...</td>
      <td>3935265.63</td>
      <td>5589608.74</td>
      <td>1871.89</td>
      <td>3011.87</td>
      <td>725.44</td>
      <td>1766</td>
      <td>210</td>
      <td>2193</td>
      <td>292977.07</td>
      <td>1897</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DORA</td>
      <td>7729</td>
      <td>14.00</td>
      <td>50.00</td>
      <td>13421.82</td>
      <td>8</td>
      <td>4</td>
      <td>0.67</td>
      <td>6</td>
      <td>75.0</td>
      <td>...</td>
      <td>2738998.00</td>
      <td>3796916.00</td>
      <td>2154.62</td>
      <td>5578.41</td>
      <td>587.28</td>
      <td>1214</td>
      <td>142</td>
      <td>1252</td>
      <td>181106.90</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>n2tstar</td>
      <td>0</td>
      <td>10.50</td>
      <td>33.33</td>
      <td>9841.04</td>
      <td>6</td>
      <td>2</td>
      <td>0.40</td>
      <td>5</td>
      <td>83.3</td>
      <td>...</td>
      <td>2347295.00</td>
      <td>3220260.00</td>
      <td>2098.47</td>
      <td>5642.54</td>
      <td>546.10</td>
      <td>1245</td>
      <td>120</td>
      <td>923</td>
      <td>160029.80</td>
      <td>1077</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 152 columns</p>
</div>




```python
concat = pd.concat([data.isnull().sum()],
          axis=1, 
          keys=['brakujące rekordy'])
print(concat.sum())
```

    brakujące rekordy    0
    dtype: int64
    


```python
described_columns=['solo_RideDistance','solo_RoadKills','solo_KillDeathRatio','duo_KillDeathRatio','solo_RoundsPlayed','solo_WinRatio','solo_LongestKill','duo_LongestKill','duo_HealsPg','duo_KillsPg','solo_WinRatio','squad_WinRatio','duo_DBNOs','duo_Revives']
print('player_name count:',data.player_name.count(),'\n')
for i in described_columns:
    print(data[i].describe().apply(lambda x: '{:2.2f}'.format(x, 'f')),'\n')
```

    player_name count: 87898 
    
    count      87898.00
    mean      102350.54
    std       153340.27
    min            0.00
    25%        19039.47
    50%        54898.48
    75%       126471.66
    max      4493014.24
    Name: solo_RideDistance, dtype: object 
    
    count    87898.00
    mean         1.55
    std          3.06
    min          0.00
    25%          0.00
    50%          1.00
    75%          2.00
    max        230.00
    Name: solo_RoadKills, dtype: object 
    
    count    87898.00
    mean         1.87
    std          1.78
    min          0.00
    25%          1.03
    50%          1.47
    75%          2.14
    max        100.00
    Name: solo_KillDeathRatio, dtype: object 
    
    count    87898.00
    mean         1.47
    std          1.35
    min          0.00
    25%          0.92
    50%          1.26
    75%          1.73
    max         86.00
    Name: duo_KillDeathRatio, dtype: object 
    
    count    87898.00
    mean        79.28
    std         96.95
    min          1.00
    25%         17.00
    50%         48.00
    75%        105.00
    max       1681.00
    Name: solo_RoundsPlayed, dtype: object 
    
    count    87898.00
    mean         5.02
    std         10.08
    min          0.00
    25%          0.00
    50%          2.00
    75%          5.56
    max        100.00
    Name: solo_WinRatio, dtype: object 
    
    count    87898.00
    mean       264.08
    std        125.82
    min          0.00
    25%        191.68
    50%        258.23
    75%        329.37
    max       4694.11
    Name: solo_LongestKill, dtype: object 
    
    count    87898.00
    mean       320.41
    std        170.87
    min          0.00
    25%        233.01
    50%        300.80
    75%        379.52
    max       5738.92
    Name: duo_LongestKill, dtype: object 
    
    count    87898.00
    mean         1.75
    std          0.73
    min          0.00
    25%          1.28
    50%          1.65
    75%          2.09
    max         17.00
    Name: duo_HealsPg, dtype: object 
    
    count    87898.00
    mean         1.36
    std          0.82
    min          0.00
    25%          0.90
    50%          1.22
    75%          1.63
    max         30.00
    Name: duo_KillsPg, dtype: object 
    
    count    87898.00
    mean         5.02
    std         10.08
    min          0.00
    25%          0.00
    50%          2.00
    75%          5.56
    max        100.00
    Name: solo_WinRatio, dtype: object 
    
    count    87898.00
    mean         6.30
    std          5.58
    min          0.00
    25%          3.09
    50%          4.98
    75%          7.81
    max        100.00
    Name: squad_WinRatio, dtype: object 
    
    count    87898.00
    mean        89.40
    std         80.44
    min          0.00
    25%         36.00
    50%         69.00
    75%        119.00
    max       1301.00
    Name: duo_DBNOs, dtype: object 
    
    count    87898.00
    mean        17.56
    std         15.33
    min          0.00
    25%          7.00
    50%         14.00
    75%         24.00
    max        190.00
    Name: duo_Revives, dtype: object 
    
    


```python
KolmogorovSmirnov = [data.solo_KillDeathRatio,data.duo_KillDeathRatio,data.duo_HealsPg,data.duo_KillsPg] #sprawdzanie czy to rozkład normalny, potrzebna informacja do zmiannych używanych w teście różnic
for i in KolmogorovSmirnov: #dla każdej stat oddzielnie
    a,b = st.kstest(i,'norm')
    print(i.name,'stat: {:2.1f}'.format(a),'pvalue:',b)
```

    solo_KillDeathRatio stat: 0.7 pvalue: 0.0
    duo_KillDeathRatio stat: 0.7 pvalue: 0.0
    duo_HealsPg stat: 0.7 pvalue: 0.0
    duo_KillsPg stat: 0.7 pvalue: 0.0
    


```python
%matplotlib inline
```

## 1. Jaki jest związek między przebytą odległością przez gracza w pojeździe (solo_RideDistance), a ilością zabójstw poprzez potrącenie (solo_RoadKills) w trybie jednoosobowym?


```python
correlation, pvalue = st.pearsonr(data['solo_RideDistance'].values, data['solo_RoadKills'].values)

print('Poziom istotności: ', pvalue)
print('Poziom korelacji:', '{:2.2f}'.format(correlation))
```

    Poziom istotności:  0.0
    Poziom korelacji: 0.71
    

## 7. Jaki jest związek między ilością wystąpień stanu DBNO (Down But Not Out) (duo_DBNOs) w trybie dwuosobowym, a ilością wskrzeszonych towarzyszy w trybie dwuosobowym (duo_Revives)?



```python
correlation, pvalue = st.pearsonr(data['duo_DBNOs'].values, data['duo_Revives'].values)

print('Poziom istotności: ', pvalue)
print('Poziom korelacji:', '{:2.2f}'.format(correlation))
```

    Poziom istotności:  0.0
    Poziom korelacji: 0.88
    

## 3. Ze wszystkich graczy, którzy zagrali więcej niż 100 rund w trybie jednoosobowym (solo_RoundsPlayed), których dziesięciu (player_name) ma najwyższy procent wygranych w trybie jednoosobowym (solo_WinRatio)? W


```python
over_100_rounds = data.loc[data.solo_RoundsPlayed > 100, ['player_name', 'solo_WinRatio']]
over_100_rounds = over_100_rounds.sort_values('solo_WinRatio', ascending=False)

top10_over_100_rounds = over_100_rounds.head(10)
print(top10_over_100_rounds)
```

                player_name  solo_WinRatio
    752    The_Venom_Inside          36.64
    1070         TGP_wahaha          31.82
    287               mjyoh          29.13
    146            denahuen          28.14
    856               Scoom          27.43
    545          IamChappie          27.10
    25389           ckmahzy          26.63
    1313           outc1der          26.50
    3045            AinsNey          25.16
    1080              benq5          24.43
    


```python
plt.figure();
ax = top10_over_100_rounds.plot(x='player_name', y='solo_WinRatio', kind='bar', fontsize=10, rot=20, legend=False, figsize=(10,6))
plt.xlabel('player_name [-]', fontsize=11)
plt.ylabel('solo_WinRatio [-]', fontsize=11)
plt.title('Najwyższy procent wygranych w trybie jednoosobowym', fontsize=11)
plt.ylim([0, 40])
for p in ax.patches:
    ax.annotate('{:2.2f}%'.format(p.get_height()), (p.get_x() - 0.03, p.get_height() + 0.6))
plt.tight_layout()

plt.savefig('top_10_over_100_rounds.png')
```


    <Figure size 432x288 with 0 Axes>



![png](output_13_1.png)


## 4. W którym z trybów (jednoosobowy i dwuosobowy) gracze osiągają najwyższy średni wynik długości śmiertelnego strzału (LongestKill) W


```python
LongestKill_solo_vs_duo = data[['solo_LongestKill', 'duo_LongestKill']]
LongestKill_solo_vs_duo = LongestKill_solo_vs_duo.mean()
LongestKill_solo_vs_duo.apply(lambda x: '{:2.2f}'.format(x, 'f'))
```




    solo_LongestKill    264.08
    duo_LongestKill     320.41
    dtype: object




```python
wykres = LongestKill_solo_vs_duo.plot.pie(autopct='%.2f % %', fontsize=11, figsize=(10, 10))
plt.legend(loc='upper left')
plt.title('Wykres porównujący średni wynik odległości śmiertelnego strzału dla tybu jednoosobowego i dwuosobowego')
plt.ylabel('')
plt.tight_layout()
plt.savefig('LongestKill_solo_vs_duo.png')
```


![png](output_16_0.png)


## 6. Jak wielu graczy, którzy mają współczynnik wygranych w trybie jednoosobowym poniżej 10 % (solo_WinRatio), mają współczynnik wygranych w trybie drużynowym powyżej 35% (squad_WinRatio). W


```python
WinRatio_solo_squad = data.loc[data.solo_WinRatio < 10 , ['solo_WinRatio','squad_WinRatio']]
WinRatio_solo_squad = WinRatio_solo_squad.loc[WinRatio_solo_squad.squad_WinRatio > 35 , ['solo_WinRatio','squad_WinRatio']]
WinRatio_solo_squad = WinRatio_solo_squad.count()
WinRatio_solo_squad['Reszta'] = data.solo_WinRatio.count()- WinRatio_solo_squad.solo_WinRatio
del WinRatio_solo_squad['squad_WinRatio']
WinRatio_solo_squad = WinRatio_solo_squad.rename({'solo_WinRatio': 'WinRatio_solo_squad_count'})
print(WinRatio_solo_squad)
```

    WinRatio_solo_squad_count      208
    Reszta                       87690
    dtype: int64
    


```python
wykres = WinRatio_solo_squad.plot.pie(autopct='%.2f % %', fontsize=11, figsize=(9, 9), explode=[0.1,0], startangle=60)
plt.legend(loc='upper left')
plt.title('Wykres porównujący liczbę graczy spełniających wymagania w stosunku do całej reszty graczy')
plt.ylabel('')
plt.tight_layout()
plt.savefig('chosenones_vs_rest.png')
```


![png](output_19_0.png)


## 2. Czy jest różnica między wartością statystyki KDRatio w trybie solo (solo_KillDeathRatio), a w trybie duo?(duo_KillDeathRatio) W


```python
statistic, pvalue = st.mannwhitneyu(data.solo_KillDeathRatio,data.duo_KillDeathRatio)
print('Poziom istotności: ', pvalue)
print('Poziom różnic:', statistic)
```

    Poziom istotności:  0.0
    Poziom różnic: 3172985334.5
    


```python
plt.figure(1 ,figsize=(10 ,10))
plt.tight_layout()
plt.xlabel('solo_KillDeathRatio [-]', fontsize=11)
plt.ylabel('duo_KillDeathRatio [-]', fontsize=11)
plt.title('Wykres zależności między KDRatio w trybie jedno i dwuosobowym', fontsize=11)
plt.scatter(data.solo_KillDeathRatio,data.duo_KillDeathRatio)
plt.tight_layout()
plt.savefig('KDRatio_solo_vs_duo.png')
```


![png](output_22_0.png)


## 5. Czy jest różnica między średnią liczbą uleczonych współtowarzyszy w trybie dwuosobowym (duo_HealsPg) a liczbą zabójstw w trybie dwuosobowym (duo_KillsPg)?


```python
statistic, pvalue = st.mannwhitneyu(data.duo_HealsPg,data.duo_KillsPg)
print('Poziom istotności: ', pvalue)
print('Poziom różnic:', statistic)
```

    Poziom istotności:  0.0
    Poziom różnic: 2373493308.0
    
