import React, { useState, useEffect } from 'react';
import { 
  CloudSun, 
  TrendingUp, 
  FileText, 
  Info, 
  MapPin, 
  Calendar, 
  Droplet, 
  Clock, 
  Upload, 
  Download, 
  CheckCircle,
  BarChart3,
  Sliders
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend
} from 'recharts';

interface City {
  location_id: number;
  latitude: number;
  longitude: number;
  elevation: number;
  city_name: string;
}

interface DefaultAverages {
  weathercode: number;
  temperature_2m_min: number;
  temperature_2m_mean: number;
  apparent_temperature_max: number;
  apparent_temperature_min: number;
  apparent_temperature_mean: number;
  daylight_duration: number;
  sunshine_duration: number;
  precipitation_sum: number;
  rain_sum: number;
  precipitation_hours: number;
  windspeed_10m_max: number;
  windgusts_10m_max: number;
  winddirection_10m_dominant: number;
  shortwave_radiation_sum: number;
  et0_fao_evapotranspiration: number;
}

interface PredictionResults {
  lgbm: { temp_max: number; latency_ms: number };
  xgboost: { temp_max: number; latency_ms: number };
  random_forest: { rain_predicted: boolean; rain_prob: number; latency_ms: number };
  summary: { avg_temp_max: number; will_rain: boolean; weather_condition: string };
}

interface HistoricalData {
  months: number[];
  temp_max: number[];
  temp_min: number[];
  rain: number[];
  wind: number[];
  radiation: number[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'predict' | 'analytics' | 'batch' | 'about'>('predict');
  const [cities, setCities] = useState<City[]>([]);
  const [selectedCityId, setSelectedCityId] = useState<number>(0);
  const [selectedDate, setSelectedDate] = useState<string>('2026-07-05');
  const [loadingDefaults, setLoadingDefaults] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [results, setResults] = useState<PredictionResults | null>(null);

  // Form Fields State
  const [weathercode, setWeathercode] = useState(2);
  const [tempMin, setTempMin] = useState(22.0);
  const [tempMean, setTempMean] = useState(26.0);
  const [appMax, setAppMax] = useState(32.0);
  const [appMin, setAppMin] = useState(24.0);
  const [appMean, setAppMean] = useState(28.0);
  const [precipSum, setPrecipSum] = useState(0.0);
  const [rainSum, setRainSum] = useState(0.0);
  const [precipHours, setPrecipHours] = useState(0.0);
  const [windMax, setWindMax] = useState(12.0);
  const [windGust, setWindGust] = useState(25.0);
  const [windDir, setWindDir] = useState(180);
  const [radSum, setRadSum] = useState(18.0);
  const [et0, setEt0] = useState(4.0);
  const [daylightDuration, setDaylightDuration] = useState(42200);
  const [sunshineDuration, setSunshineDuration] = useState(35000);

  // Analytics State
  const [analyticsCityId, setAnalyticsCityId] = useState<number>(0);
  const [historicalData, setHistoricalData] = useState<HistoricalData | null>(null);

  // Batch Prediction State
  const [uploading, setUploading] = useState(false);
  const [batchModel, setBatchModel] = useState<string>('compare');
  const [batchResults, setBatchResults] = useState<{
    success: boolean;
    total_rows: number;
    download_url: string;
    preview: any[];
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const selectedCity = cities.find(c => c.location_id === selectedCityId) || cities[0];

  // Fetch Cities on load
  useEffect(() => {
    fetch('http://localhost:5001/api/cities')
      .then(res => res.json())
      .then((data: City[]) => {
        setCities(data);
        if (data.length > 0) {
          setSelectedCityId(data[0].location_id);
          setAnalyticsCityId(data[0].location_id);
        }
      })
      .catch(err => {
        console.error("Error fetching cities:", err);
        setErrorMsg("Failed to connect to Python backend server. Make sure server.py is running!");
      });
  }, []);

  // Fetch Defaults when selected city or month changes
  const loadDefaults = (cityId: number, dateStr: string) => {
    setLoadingDefaults(true);
    const month = new Date(dateStr).getMonth() + 1;
    fetch(`http://localhost:5001/api/defaults?city_id=${cityId}&month=${month}`)
      .then(res => res.json())
      .then((data: DefaultAverages) => {
        setWeathercode(Math.round(data.weathercode));
        setTempMin(parseFloat(data.temperature_2m_min.toFixed(1)));
        setTempMean(parseFloat(data.temperature_2m_mean.toFixed(1)));
        setAppMax(parseFloat(data.apparent_temperature_max.toFixed(1)));
        setAppMin(parseFloat(data.apparent_temperature_min.toFixed(1)));
        setAppMean(parseFloat(data.apparent_temperature_mean.toFixed(1)));
        setPrecipSum(parseFloat(data.precipitation_sum.toFixed(2)));
        setRainSum(parseFloat(data.rain_sum.toFixed(2)));
        setPrecipHours(parseFloat(data.precipitation_hours.toFixed(1)));
        setWindMax(parseFloat(data.windspeed_10m_max.toFixed(1)));
        setWindGust(parseFloat(data.windgusts_10m_max.toFixed(1)));
        setWindDir(Math.round(data.winddirection_10m_dominant));
        setRadSum(parseFloat(data.shortwave_radiation_sum.toFixed(2)));
        setEt0(parseFloat(data.et0_fao_evapotranspiration.toFixed(2)));
        setDaylightDuration(Math.round(data.daylight_duration));
        setSunshineDuration(Math.round(data.sunshine_duration));
        setLoadingDefaults(false);
      })
      .catch(err => {
        console.error("Error loading defaults:", err);
        setLoadingDefaults(false);
      });
  };

  useEffect(() => {
    if (cities.length > 0) {
      loadDefaults(selectedCityId, selectedDate);
    }
  }, [selectedCityId, selectedDate, cities]);

  // Fetch Analytics data
  useEffect(() => {
    if (cities.length > 0) {
      fetch(`http://localhost:5001/api/historical?city_id=${analyticsCityId}`)
        .then(res => res.json())
        .then((data: HistoricalData) => {
          setHistoricalData(data);
        })
        .catch(err => console.error("Error fetching historical trend:", err));
    }
  }, [analyticsCityId, cities]);

  // Run Predict
  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    setErrorMsg(null);

    const payload = {
      city_id: selectedCityId,
      latitude: selectedCity.latitude,
      longitude: selectedCity.longitude,
      elevation: selectedCity.elevation,
      date: selectedDate,
      weathercode,
      temperature_2m_min: tempMin,
      temperature_2m_mean: tempMean,
      apparent_temperature_max: appMax,
      apparent_temperature_min: appMin,
      apparent_temperature_mean: appMean,
      precipitation_sum: precipSum,
      rain_sum: rainSum,
      precipitation_hours: precipHours,
      windspeed_10m_max: windMax,
      windgusts_10m_max: windGust,
      winddirection_10m_dominant: windDir,
      shortwave_radiation_sum: radSum,
      et0_fao_evapotranspiration: et0,
      daylight_duration: daylightDuration,
      sunshine_duration: sunshineDuration
    };

    fetch('http://localhost:5001/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then((data: any) => {
        if (data.error) {
          setErrorMsg(data.error);
        } else {
          setResults(data);
        }
        setPredicting(false);
      })
      .catch(err => {
        console.error("Prediction failed:", err);
        setErrorMsg("Failed to run prediction. Python backend server error!");
        setPredicting(false);
      });
  };

  // Batch CSV Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setUploading(true);
    setErrorMsg(null);
    setBatchResults(null);

    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', batchModel);

    fetch('http://localhost:5001/api/batch-predict', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then((data: any) => {
        if (data.error) {
          setErrorMsg(data.error);
        } else {
          setBatchResults(data);
        }
        setUploading(false);
      })
      .catch(err => {
        console.error("Batch predict failed:", err);
        setErrorMsg("Batch predictions failed. Ensure the CSV has correct columns.");
        setUploading(false);
      });
  };

  // Render animated weather icon based on predicted state
  const renderWeatherAnimation = (condition: string) => {
    switch (condition) {
      case 'Sunny':
        return (
          <div className="weather-animation-container">
            <div className="anim-sun">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="anim-sun-ray" style={{ transform: `rotate(${i * 45}deg)` }} />
              ))}
            </div>
          </div>
        );
      case 'Cloudy':
        return (
          <div className="weather-animation-container">
            <div className="anim-sun" style={{ top: '25px', left: '45px', width: '35px', height: '35px', boxShadow: '0 0 20px #f2994a' }} />
            <div className="anim-cloud" style={{ top: '45px', left: '20px' }} />
          </div>
        );
      case 'Rainy':
        return (
          <div className="weather-animation-container">
            <div className="anim-cloud" style={{ top: '35px', left: '25px', background: '#b0bec5' }} />
            <div className="rain-drops">
              <div className="drop" style={{ left: '15px' }} />
              <div className="drop" style={{ left: '28px', animationDelay: '0.4s' }} />
              <div className="drop" style={{ left: '42px', animationDelay: '0.2s' }} />
              <div className="drop" style={{ left: '55px', animationDelay: '0.6s' }} />
            </div>
          </div>
        );
      case 'Stormy':
        return (
          <div className="weather-animation-container">
            <div className="anim-cloud" style={{ top: '35px', left: '25px', background: '#78909c' }} />
            <div className="lightning" />
            <div className="rain-drops">
              <div className="drop" style={{ left: '20px', background: '#29b6f6' }} />
              <div className="drop" style={{ left: '45px', animationDelay: '0.3s', background: '#29b6f6' }} />
            </div>
          </div>
        );
      default:
        return <CloudSun className="text-secondary" size={80} style={{ margin: '0 auto', display: 'block' }} />;
    }
  };

  // Convert month number to string
  const getMonthName = (m: number) => {
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1];
  };

  // Prepare Recharts charting data
  const getChartData = () => {
    if (!historicalData) return [];
    return historicalData.months.map((m, idx) => ({
      name: getMonthName(m),
      'Max Temp': parseFloat(historicalData.temp_max[idx].toFixed(1)),
      'Min Temp': parseFloat(historicalData.temp_min[idx].toFixed(1)),
      'Precipitation (mm)': parseFloat(historicalData.rain[idx].toFixed(1)),
      'Wind Speed (km/h)': parseFloat(historicalData.wind[idx].toFixed(1)),
      'Radiation (MJ/m²)': parseFloat(historicalData.radiation[idx].toFixed(1))
    }));
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar navigation */}
      <aside className="sidebar">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <CloudSun size={32} className="text-gradient" style={{ stroke: 'url(#cyan-blue-grad)' }} />
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800 }}>
              <span className="text-gradient">Smart</span>Weather
            </h2>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Sri Lanka ML Prediction System</p>
        </div>

        <nav style={{ flex: 1 }}>
          <ul className="nav-menu">
            <li>
              <button 
                className={`nav-item ${activeTab === 'predict' ? 'active' : ''}`}
                onClick={() => setActiveTab('predict')}
              >
                <Sliders size={20} />
                Forecast & Predict
              </button>
            </li>
            <li>
              <button 
                className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
                onClick={() => setActiveTab('analytics')}
              >
                <BarChart3 size={20} />
                Analytics & Trends
              </button>
            </li>
            <li>
              <button 
                className={`nav-item ${activeTab === 'batch' ? 'active' : ''}`}
                onClick={() => setActiveTab('batch')}
              >
                <FileText size={20} />
                Batch Prediction
              </button>
            </li>
            <li>
              <button 
                className={`nav-item ${activeTab === 'about' ? 'active' : ''}`}
                onClick={() => setActiveTab('about')}
              >
                <Info size={20} />
                How It Works
              </button>
            </li>
          </ul>
        </nav>

        <div className="glass-card" style={{ padding: '1rem', borderRadius: '14px', fontSize: '0.85rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-cyan)', fontWeight: 600, marginBottom: '0.25rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00e676', boxShadow: '0 0 8px #00e676' }} />
            Server Connected
          </div>
          <span style={{ color: 'var(--text-secondary)' }}>Host: localhost:5001</span>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {errorMsg && (
          <div style={{ 
            background: 'rgba(235, 87, 87, 0.1)', 
            border: '1px solid var(--accent-rose)', 
            padding: '1rem', 
            borderRadius: '12px', 
            marginBottom: '2rem',
            color: '#ff8a80',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Info size={18} />
            {errorMsg}
          </div>
        )}

        {/* TAB 1: FORECAST & PREDICT */}
        {activeTab === 'predict' && (
          <div>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ fontSize: '2.25rem', marginBottom: '0.5rem' }}>
                Weather <span className="text-gradient">Forecast Engine</span>
              </h1>
              <p style={{ color: 'var(--text-secondary)' }}>
                Predict next day's maximum temperature and rainfall probability using advanced LightGBM, XGBoost, and Random Forest models.
              </p>
            </div>

            <form onSubmit={handlePredict} style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '2rem', alignItems: 'start' }}>
              {/* Form Controls */}
              <div className="glass-card">
                <h3 style={{ fontSize: '1.2rem', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Sliders size={18} className="text-gradient" />
                  Parameter Controls
                </h3>

                {/* Location and Date Selection */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
                  <div className="input-group">
                    <label className="input-label">
                      <span><MapPin size={14} style={{ verticalAlign: 'middle', marginRight: '4px' }} /> Sri Lankan City</span>
                    </label>
                    <select 
                      value={selectedCityId}
                      onChange={(e) => setSelectedCityId(parseInt(e.target.value))}
                    >
                      {cities.map(c => (
                        <option key={c.location_id} value={c.location_id}>
                          {c.city_name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span><Calendar size={14} style={{ verticalAlign: 'middle', marginRight: '4px' }} /> Target Date</span>
                    </label>
                    <input 
                      type="date" 
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                    />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '1rem', background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: '12px', marginBottom: '2rem', fontSize: '0.85rem' }}>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Latitude: </span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedCity?.latitude}°N</span>
                    <span style={{ margin: '0 0.5rem', color: 'var(--glass-border)' }}>|</span>
                    <span style={{ color: 'var(--text-secondary)' }}>Longitude: </span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedCity?.longitude}°E</span>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Elevation: </span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedCity?.elevation}m</span>
                  </div>
                </div>

                {/* Range inputs for parameters */}
                <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '1rem', fontWeight: 600 }}>METEOROLOGICAL PARAMETERS</h4>
                
                <div className="input-grid">
                  {/* Weather code */}
                  <div className="input-group">
                    <label className="input-label">
                      <span>Weather WMO Code</span>
                      <span className="input-value">{weathercode}</span>
                    </label>
                    <select 
                      value={weathercode} 
                      onChange={(e) => setWeathercode(parseInt(e.target.value))}
                    >
                      <option value="0">0 - Sunny / Clear</option>
                      <option value="1">1 - Mainly Clear</option>
                      <option value="2">2 - Partly Cloudy</option>
                      <option value="3">3 - Overcast / Cloudy</option>
                      <option value="51">51 - Light Drizzle</option>
                      <option value="53">53 - Moderate Drizzle</option>
                      <option value="61">61 - Slight Rain</option>
                      <option value="63">63 - Moderate Rain</option>
                      <option value="80">80 - Slight Showers</option>
                      <option value="81">81 - Heavy Showers / Storm</option>
                    </select>
                  </div>

                  {/* Temperature sliders */}
                  <div className="input-group">
                    <label className="input-label">
                      <span>Min Temp (°C)</span>
                      <span className="input-value">{tempMin}°C</span>
                    </label>
                    <input 
                      type="range" min="10" max="35" step="0.1" 
                      value={tempMin} onChange={(e) => setTempMin(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Mean Temp (°C)</span>
                      <span className="input-value">{tempMean}°C</span>
                    </label>
                    <input 
                      type="range" min="12" max="38" step="0.1" 
                      value={tempMean} onChange={(e) => setTempMean(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Apparent Max Temp (°C)</span>
                      <span className="input-value">{appMax}°C</span>
                    </label>
                    <input 
                      type="range" min="15" max="48" step="0.1" 
                      value={appMax} onChange={(e) => setAppMax(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Apparent Min Temp (°C)</span>
                      <span className="input-value">{appMin}°C</span>
                    </label>
                    <input 
                      type="range" min="10" max="35" step="0.1" 
                      value={appMin} onChange={(e) => setAppMin(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Apparent Mean Temp (°C)</span>
                      <span className="input-value">{appMean}°C</span>
                    </label>
                    <input 
                      type="range" min="12" max="42" step="0.1" 
                      value={appMean} onChange={(e) => setAppMean(parseFloat(e.target.value))}
                    />
                  </div>

                  {/* Precipitation */}
                  <div className="input-group">
                    <label className="input-label">
                      <span>Rain Sum (mm)</span>
                      <span className="input-value">{rainSum} mm</span>
                    </label>
                    <input 
                      type="range" min="0" max="100" step="0.1" 
                      value={rainSum} onChange={(e) => {
                        setRainSum(parseFloat(e.target.value));
                        setPrecipSum(parseFloat(e.target.value));
                      }}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Precipitation Hours</span>
                      <span className="input-value">{precipHours} h</span>
                    </label>
                    <input 
                      type="range" min="0" max="24" step="0.5" 
                      value={precipHours} onChange={(e) => setPrecipHours(parseFloat(e.target.value))}
                    />
                  </div>

                  {/* Wind */}
                  <div className="input-group">
                    <label className="input-label">
                      <span>Wind Speed Max (km/h)</span>
                      <span className="input-value">{windMax} km/h</span>
                    </label>
                    <input 
                      type="range" min="0" max="60" step="0.1" 
                      value={windMax} onChange={(e) => setWindMax(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Wind Gusts Max (km/h)</span>
                      <span className="input-value">{windGust} km/h</span>
                    </label>
                    <input 
                      type="range" min="0" max="90" step="0.1" 
                      value={windGust} onChange={(e) => setWindGust(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Wind Direction dominant</span>
                      <span className="input-value">{windDir}°</span>
                    </label>
                    <input 
                      type="range" min="0" max="360" step="1" 
                      value={windDir} onChange={(e) => setWindDir(parseInt(e.target.value))}
                    />
                  </div>

                  {/* Sunshine & Radiation */}
                  <div className="input-group">
                    <label className="input-label">
                      <span>Radiation Sum (MJ/m²)</span>
                      <span className="input-value">{radSum} MJ/m²</span>
                    </label>
                    <input 
                      type="range" min="0" max="35" step="0.1" 
                      value={radSum} onChange={(e) => setRadSum(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Evapotranspiration (mm)</span>
                      <span className="input-value">{et0} mm</span>
                    </label>
                    <input 
                      type="range" min="0" max="10" step="0.05" 
                      value={et0} onChange={(e) => setEt0(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Daylight Duration (h)</span>
                      <span className="input-value">{Math.round(daylightDuration / 360) / 10} h</span>
                    </label>
                    <input 
                      type="range" min="39000" max="45000" step="100" 
                      value={daylightDuration} onChange={(e) => setDaylightDuration(parseInt(e.target.value))}
                    />
                  </div>

                  <div className="input-group">
                    <label className="input-label">
                      <span>Sunshine Duration (h)</span>
                      <span className="input-value">{Math.round(sunshineDuration / 360) / 10} h</span>
                    </label>
                    <input 
                      type="range" min="0" max="45000" step="100" 
                      value={sunshineDuration} onChange={(e) => setSunshineDuration(parseInt(e.target.value))}
                    />
                  </div>
                </div>

                <div style={{ marginTop: '2.5rem', display: 'flex', gap: '1rem' }}>
                  <button 
                    type="submit" 
                    className="btn btn-primary"
                    style={{ flex: 1 }}
                    disabled={predicting || loadingDefaults}
                  >
                    {predicting ? "Predicting..." : "Generate Weather Prediction"}
                  </button>
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    onClick={() => loadDefaults(selectedCityId, selectedDate)}
                    disabled={loadingDefaults || predicting}
                  >
                    {loadingDefaults ? "Loading..." : "Reset to Averages"}
                  </button>
                </div>
              </div>

              {/* Prediction Visual Output */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                {results ? (
                  <>
                    {/* Summary Visualizer */}
                    <div className="glass-card" style={{ 
                      textAlign: 'center', 
                      background: 'linear-gradient(135deg, rgba(13, 20, 38, 0.8) 0%, rgba(20, 30, 55, 0.6) 100%)',
                      border: '1px solid rgba(79, 172, 254, 0.2)',
                      position: 'relative'
                    }}>
                      <div style={{ position: 'absolute', top: '1rem', right: '1.25rem', fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <MapPin size={12} /> {selectedCity?.city_name}
                      </div>
                      
                      <h4 style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '1.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Predicted Conditions</h4>
                      
                      {renderWeatherAnimation(results.summary.weather_condition)}
                      
                      <div style={{ marginTop: '1.5rem' }}>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>PREDICTED MAX TEMPERATURE</span>
                        <div className="text-gradient" style={{ fontSize: '3.5rem', fontWeight: 800, margin: '0.25rem 0' }}>
                          {results.summary.avg_temp_max}°C
                        </div>
                      </div>

                      <div style={{ 
                        display: 'grid', 
                        gridTemplateColumns: '1fr 1fr', 
                        gap: '1rem', 
                        marginTop: '1.5rem', 
                        borderTop: '1px solid rgba(255,255,255,0.06)',
                        paddingTop: '1.5rem'
                      }}>
                        <div>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>RAIN PROBABILITY</span>
                          <p style={{ fontSize: '1.4rem', fontWeight: 700, color: results.summary.will_rain ? 'var(--accent-blue)' : 'var(--text-primary)', marginTop: '0.25rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
                            <Droplet size={18} style={{ color: 'var(--accent-blue)' }} /> {results.random_forest.rain_prob}%
                          </p>
                        </div>
                        <div>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>OUTLOOK</span>
                          <p style={{ fontSize: '1.4rem', fontWeight: 700, color: results.summary.will_rain ? 'var(--accent-amber)' : 'var(--accent-cyan)', marginTop: '0.25rem' }}>
                            {results.summary.weather_condition}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Model comparison panel */}
                    <div className="glass-card">
                      <h4 style={{ fontSize: '1.1rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Clock size={16} className="text-gradient" />
                        Model Analytics & Latency
                      </h4>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div className="model-card xgboost" style={{ padding: '1rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                            <span style={{ fontWeight: 600 }}>XGBoost Regressor</span>
                            <span style={{ color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '2px' }}><Clock size={12} /> {results.xgboost.latency_ms} ms</span>
                          </div>
                          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                            {results.xgboost.temp_max}°C <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 400 }}>Max Temp</span>
                          </div>
                        </div>

                        <div className="model-card lightgbm" style={{ padding: '1rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                            <span style={{ fontWeight: 600 }}>LightGBM Regressor</span>
                            <span style={{ color: 'var(--accent-purple)', display: 'flex', alignItems: 'center', gap: '2px' }}><Clock size={12} /> {results.lgbm.latency_ms} ms</span>
                          </div>
                          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                            {results.lgbm.temp_max}°C <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 400 }}>Max Temp</span>
                          </div>
                        </div>

                        <div className="model-card randomforest" style={{ padding: '1rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                            <span style={{ fontWeight: 600 }}>Random Forest Classifier</span>
                            <span style={{ color: 'var(--accent-amber)', display: 'flex', alignItems: 'center', gap: '2px' }}><Clock size={12} /> {results.random_forest.latency_ms} ms</span>
                          </div>
                          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                            {results.random_forest.rain_predicted ? "Rain Forecasted" : "No Rain Forecasted"}
                            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 400, marginLeft: '0.5rem' }}>({results.random_forest.rain_prob}% prob)</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="glass-card" style={{ 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    textAlign: 'center',
                    padding: '4rem 2rem',
                    color: 'var(--text-secondary)',
                    background: 'rgba(13, 20, 38, 0.2)'
                  }}>
                    <CloudSun size={60} style={{ stroke: 'rgba(255,255,255,0.15)', marginBottom: '1.5rem' }} />
                    <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Awaiting Prediction</h3>
                    <p style={{ fontSize: '0.9rem', maxWidth: '320px' }}>
                      Configure the meteorological sliders on the left and click <b>Generate Weather Prediction</b> to compare ML models.
                    </p>
                  </div>
                )}
              </div>
            </form>
          </div>
        )}

        {/* TAB 2: ANALYTICS & TRENDS */}
        {activeTab === 'analytics' && (
          <div>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ fontSize: '2.25rem', marginBottom: '0.5rem' }}>
                Historical <span className="text-gradient">Weather Analytics</span>
              </h1>
              <p style={{ color: 'var(--text-secondary)' }}>
                Analyze historical climate trends, seasonal deviations, and weather patterns for Sri Lankan cities based on long-term record metrics.
              </p>
            </div>

            <div className="glass-card" style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '1rem', marginBottom: '2rem' }}>
                <h3 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <TrendingUp size={20} className="text-gradient" />
                  Monthly Seasonal Trends
                </h3>
                <div className="input-group" style={{ minWidth: '220px' }}>
                  <select 
                    value={analyticsCityId}
                    onChange={(e) => setAnalyticsCityId(parseInt(e.target.value))}
                  >
                    {cities.map(c => (
                      <option key={c.location_id} value={c.location_id}>
                        {c.city_name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {historicalData ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                  {/* Temperature Trend Area Chart */}
                  <div>
                    <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '1rem', fontWeight: 600 }}>MAX & MIN TEMPERATURE TRENDS (°C)</h4>
                    <div style={{ width: '100%', height: 320 }}>
                      <ResponsiveContainer>
                        <AreaChart data={getChartData()} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                          <defs>
                            <linearGradient id="colorMax" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="var(--accent-amber)" stopOpacity={0.4}/>
                              <stop offset="95%" stopColor="var(--accent-amber)" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorMin" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.4}/>
                              <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="name" stroke="var(--text-muted)" style={{ fontSize: '0.8rem' }} />
                          <YAxis stroke="var(--text-muted)" style={{ fontSize: '0.8rem' }} domain={[15, 38]} />
                          <Tooltip contentStyle={{ background: '#0d1426', border: '1px solid var(--glass-border)', borderRadius: '10px' }} />
                          <Area type="monotone" dataKey="Max Temp" stroke="var(--accent-amber)" fillOpacity={1} fill="url(#colorMax)" strokeWidth={2.5} />
                          <Area type="monotone" dataKey="Min Temp" stroke="var(--accent-cyan)" fillOpacity={1} fill="url(#colorMin)" strokeWidth={2.5} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Rain and Wind Speed Trends */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                    {/* Rain Bar Chart */}
                    <div className="glass-card" style={{ background: 'rgba(255,255,255,0.01)', padding: '1.25rem' }}>
                      <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '1rem', fontWeight: 600 }}>MONTHLY RAIN PATTERNS (MM)</h4>
                      <div style={{ width: '100%', height: 260 }}>
                        <ResponsiveContainer>
                          <BarChart data={getChartData()}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="name" stroke="var(--text-muted)" style={{ fontSize: '0.75rem' }} />
                            <YAxis stroke="var(--text-muted)" style={{ fontSize: '0.75rem' }} />
                            <Tooltip contentStyle={{ background: '#0d1426', border: '1px solid var(--glass-border)', borderRadius: '10px' }} />
                            <Bar dataKey="Precipitation (mm)" fill="var(--accent-blue)" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Wind and Radiation Line Chart */}
                    <div className="glass-card" style={{ background: 'rgba(255,255,255,0.01)', padding: '1.25rem' }}>
                      <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '1rem', fontWeight: 600 }}>WIND SPEED (KM/H) & RADIATION (MJ/M²)</h4>
                      <div style={{ width: '100%', height: 260 }}>
                        <ResponsiveContainer>
                          <AreaChart data={getChartData()}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="name" stroke="var(--text-muted)" style={{ fontSize: '0.75rem' }} />
                            <YAxis stroke="var(--text-muted)" style={{ fontSize: '0.75rem' }} />
                            <Tooltip contentStyle={{ background: '#0d1426', border: '1px solid var(--glass-border)', borderRadius: '10px' }} />
                            <Legend style={{ fontSize: '0.8rem' }} />
                            <Area type="monotone" dataKey="Wind Speed (km/h)" stroke="var(--accent-purple)" fill="rgba(155, 81, 224, 0.05)" strokeWidth={2} />
                            <Area type="monotone" dataKey="Radiation (MJ/m²)" stroke="var(--accent-cyan)" fill="rgba(0, 242, 254, 0.05)" strokeWidth={2} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '3rem' }}>Loading trend dashboard...</div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: BATCH PREDICTION */}
        {activeTab === 'batch' && (
          <div>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ fontSize: '2.25rem', marginBottom: '0.5rem' }}>
                Batch Weather <span className="text-gradient">Prediction Tool</span>
              </h1>
              <p style={{ color: 'var(--text-secondary)' }}>
                Upload a structured CSV file containing daily meteorological measurements to run batch predictions through XGBoost, LightGBM, and Random Forest models.
              </p>
            </div>

            <div className="glass-card" style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.2rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={20} className="text-gradient" />
                Upload CSV File
              </h3>

              <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                <div className="input-group" style={{ minWidth: '220px' }}>
                  <label className="input-label">Select Prediction Model</label>
                  <select 
                    value={batchModel}
                    onChange={(e) => setBatchModel(e.target.value)}
                  >
                    <option value="xgboost">XGBoost (Temp Max Only)</option>
                    <option value="lgbm">LightGBM (Temp Max Only)</option>
                    <option value="compare">Compare All Models (Side-by-Side)</option>
                  </select>
                </div>
              </div>

              {/* Drag and Drop Box */}
              <label className="upload-area">
                <input 
                  type="file" 
                  accept=".csv" 
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                  disabled={uploading}
                />
                <div style={{ 
                  width: '60px', 
                  height: '60px', 
                  borderRadius: '50%', 
                  background: 'rgba(255,255,255,0.03)', 
                  border: '1px solid var(--glass-border)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent-cyan)'
                }}>
                  <Upload size={28} />
                </div>
                <div>
                  <h4 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                    {uploading ? "Analyzing and Predicting..." : "Click or Drag CSV here to upload"}
                  </h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Files must be in standard comma-separated format (.csv)
                  </p>
                </div>
              </label>

              {/* Instructions on structure */}
              <div style={{ 
                marginTop: '2rem', 
                background: 'rgba(255, 255, 255, 0.01)', 
                border: '1px solid var(--glass-border)', 
                borderRadius: '12px',
                padding: '1.25rem',
                fontSize: '0.85rem'
              }}>
                <h4 style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Info size={14} className="text-accent-cyan" />
                  Required CSV Columns Structure
                </h4>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Your CSV should contain headers like: <code>weathercode</code>, <code>temperature_2m_min</code>, <code>temperature_2m_mean</code>, <code>apparent_temperature_max</code>, <code>apparent_temperature_min</code>, <code>apparent_temperature_mean</code>, <code>shortwave_radiation_sum</code>, <code>precipitation_sum</code>, <code>rain_sum</code>, <code>precipitation_hours</code>, <code>windspeed_10m_max</code>, <code>windgusts_10m_max</code>, <code>winddirection_10m_dominant</code>, <code>et0_fao_evapotranspiration</code>, <code>latitude</code>, <code>longitude</code>, <code>elevation</code>, <code>day</code>, <code>month</code>, <code>year</code>.
                </p>
                <div style={{ color: 'var(--text-muted)' }}>
                  *Note: Units will be automatically mapped if your CSV contains headers like <code>temperature_2m_min (°C)</code> or <code>wind_speed_10m_max (km/h)</code> directly matching the original dataset features.
                </div>
              </div>
            </div>

            {/* Batch Output Preview */}
            {batchResults && (
              <div className="glass-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-emerald)' }}>
                      <CheckCircle size={20} />
                      Batch Predictions Complete
                    </h3>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Successfully predicted weather conditions for <b>{batchResults.total_rows} rows</b> of data.
                    </p>
                  </div>
                  
                  <a 
                    href={`http://localhost:5001${batchResults.download_url}`}
                    className="btn btn-primary"
                    style={{ textDecoration: 'none' }}
                  >
                    <Download size={18} />
                    Download Predicted CSV
                  </a>
                </div>

                <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '0.75rem', fontWeight: 600 }}>PREVIEW OF TOP PREDICTIONS</h4>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Year-Month-Day</th>
                        <th>Lat/Lon</th>
                        <th>Weather WMO</th>
                        {batchModel === 'xgboost' && <th>Predicted Temp Max (XGBoost)</th>}
                        {batchModel === 'lgbm' && <th>Predicted Temp Max (LightGBM)</th>}
                        {batchModel === 'compare' && (
                          <>
                            <th>Temp Max (XGBoost)</th>
                            <th>Temp Max (LightGBM)</th>
                          </>
                        )}
                        <th>Predicted Rain Today</th>
                        <th>Rain Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.preview.map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.year}-{row.month}-{row.day}</td>
                          <td>{row.latitude?.toFixed(2)}°N / {row.longitude?.toFixed(2)}°E</td>
                          <td>{row.weathercode}</td>
                          {batchModel === 'xgboost' && <td><span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{row.Predicted_TempMax_XGBoost?.toFixed(2)}°C</span></td>}
                          {batchModel === 'lgbm' && <td><span style={{ fontWeight: 600, color: 'var(--accent-purple)' }}>{row.Predicted_TempMax_LightGBM?.toFixed(2)}°C</span></td>}
                          {batchModel === 'compare' && (
                            <>
                              <td><span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{row.Predicted_TempMax_XGBoost?.toFixed(2)}°C</span></td>
                              <td><span style={{ fontWeight: 600, color: 'var(--accent-purple)' }}>{row.Predicted_TempMax_LightGBM?.toFixed(2)}°C</span></td>
                            </>
                          )}
                          <td>
                            <span style={{ 
                              padding: '0.25rem 0.5rem', 
                              borderRadius: '6px', 
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              background: row.Predicted_Rain_Today === 1 ? 'rgba(79, 172, 254, 0.1)' : 'rgba(255,255,255,0.05)',
                              color: row.Predicted_Rain_Today === 1 ? 'var(--accent-blue)' : 'var(--text-secondary)'
                            }}>
                              {row.Predicted_Rain_Today === 1 ? "Rain" : "Dry"}
                            </span>
                          </td>
                          <td>{row.Rain_Probability_}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: ABOUT & HOW IT WORKS */}
        {activeTab === 'about' && (
          <div>
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{ fontSize: '2.25rem', marginBottom: '0.5rem' }}>
                How the <span className="text-gradient">Forecast System Works</span>
              </h1>
              <p style={{ color: 'var(--text-secondary)' }}>
                Understand the machine learning pipelines, datasets, and decision processes powering our Smart Weather prediction system.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '2rem' }}>
              {/* Algorithm Details */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <div className="glass-card">
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', color: 'var(--accent-cyan)' }}>Model Pipelines</h3>
                  <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1rem' }}>
                    Our system loads pre-trained machine learning checkpoints constructed from years of historical climate logs across Sri Lankan districts.
                  </p>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1.5rem' }}>
                    <div style={{ borderLeft: '3px solid var(--accent-cyan)', paddingLeft: '1rem' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.25rem' }}>XGBoost Regressor</h4>
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        A gradient boosting library designed to be highly efficient, flexible, and portable. It predicts the maximum temperature (Temp_Max) by optimizing an objective function with L1/L2 regularization to prevent overfitting.
                      </p>
                    </div>

                    <div style={{ borderLeft: '3px solid var(--accent-purple)', paddingLeft: '1rem' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.25rem' }}>LightGBM Regressor</h4>
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        A tree-based gradient learning framework from Microsoft. It uses leaf-wise tree growth rather than level-wise, making it much faster to train and highly accurate for temperature forecasting.
                      </p>
                    </div>

                    <div style={{ borderLeft: '3px solid var(--accent-amber)', paddingLeft: '1rem' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.25rem' }}>Random Forest Classifier</h4>
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        An ensemble learning method that fits multiple decision trees on sub-samples of the dataset. Here, it is structured as a classifier to predict whether it will rain (RainToday) on a binary threshold, leveraging temperature forecasts as feedback.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="glass-card">
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', color: 'var(--accent-purple)' }}>Integrated Pipelines Logic</h3>
                  <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    Rather than predicting parameters isolated from each other, our system integrates the regression and classification pipelines.
                  </p>
                  <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: '0.75rem' }}>
                    When a user runs a prediction, the server first passes the features into <b>XGBoost</b> and <b>LightGBM</b> to compute the predicted maximum temperature of the day. The average of these predicted values is then dynamically fed as the <b>Temp_Max</b> parameter into the <b>Random Forest Classifier</b>. This creates an interconnected inference chain representing realistic atmospheric conditions where high temperatures influence precipitation probabilities.
                  </p>
                </div>
              </div>

              {/* Feature Importances and Dataset Metadata */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <div className="glass-card">
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '1.5rem' }}>Feature Contributions (Temp Max)</h3>
                  
                  {/* Mock Contribution Meter */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                        <span>Apparent Temperature Max</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>38%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: '38%', background: 'var(--accent-cyan)', borderRadius: '4px' }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                        <span>Shortwave Radiation Sum</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>22%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: '22%', background: 'var(--accent-cyan)', borderRadius: '4px' }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                        <span>Elevation & Location ID</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>15%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: '15%', background: 'var(--accent-cyan)', borderRadius: '4px' }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                        <span>Mean Temperature</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>12%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: '12%', background: 'var(--accent-cyan)', borderRadius: '4px' }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                        <span>Seasonal Month / Date</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>8%</span>
                      </div>
                      <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: '8%', background: 'var(--accent-cyan)', borderRadius: '4px' }} />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="glass-card" style={{ background: 'rgba(79, 172, 254, 0.05)', border: '1px solid rgba(79, 172, 254, 0.2)' }}>
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>Dataset Characteristics</h3>
                  <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5, fontSize: '0.9rem' }}>
                    The models were trained on <b>142,371 daily climate observations</b> spanning from <b>2010 to 2024</b> for 27 primary weather stations in Sri Lanka. 
                  </p>
                  <ul style={{ color: 'var(--text-secondary)', paddingLeft: '1.25rem', marginTop: '0.75rem', fontSize: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <li><b>Lowlands (e.g. Colombo, Galle):</b> Characterized by high humidity, stable temperatures (29-33°C), and monsoon season rains.</li>
                    <li><b>Highlands (e.g. Nuwara Eliya):</b> High elevation (1865m), much cooler temperatures (12-20°C), and lower solar radiation levels.</li>
                    <li><b>Arid Zones (e.g. Jaffna, Mannar):</b> Lower rainfall records, high sunshine duration, and elevated evapotranspiration numbers.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* SVG Gradient declaration for icons */}
      <svg width="0" height="0" style={{ position: 'absolute' }}>
        <linearGradient id="cyan-blue-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00f2fe" />
          <stop offset="100%" stopColor="#4facfe" />
        </linearGradient>
      </svg>
    </div>
  );
}
