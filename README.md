# Symbol

Symbol allows you to analyze anomalous stock market movement.   

### Kodak lookin' a little strange...?

![kodak ticker](https://github.com/estepmax/symbol/blob/master/screenshots/kodak_ticker.PNG)

## Usage

### Plotting four hidden states for volume indicator
```python
sym = Symbol("KODK")
sym.get_indicators(start="2020-07-27",end="2020-07-31",interval="1m",prepost=True)
sym.fit(indtype=sym.volume())
sym.plot_hidden_states(indtype=sym.volume())
```

![hidden states](https://github.com/estepmax/symbol/blob/master/screenshots/hidden_states.png)

### Anomalous volume indicator
```python
sym.anomaly(indtype=sym.volume(),label_anomalous=False)
sym.anomaly_distribution(sym.volume())
```
![anomalous volume](https://github.com/estepmax/symbol/blob/master/screenshots/anomalies.png)

![anomalous dist](https://github.com/estepmax/symbol/blob/master/screenshots/anomdist.png)

### TODO
- Add the ability to query upcoming earnings & related news for each symbol
- Some sort of way to perform correlation analysis between symbols  
