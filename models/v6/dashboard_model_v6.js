/**
 * Model v6 Dashboard Implementation
 *
 * JavaScript implementation for client-side weather-adjusted predictions.
 * Supports elevation and marine layer features.
 *
 * Usage:
 *   import { PredictionModel } from './dashboard_model_v6.js';
 *   const model = new PredictionModel(paramsJson);
 *   const result = model.predict('SF', { temp_f: 65, rhum: 75, ... }, 'runs');
 */

// Marine layer parks - Pacific coast stadiums with fog influence
const MARINE_LAYER_PARKS = ['SF', 'OAK', 'SD', 'LAA', 'LAD', 'SEA'];

// Park display names
const PARK_NAMES = {
  ARI: 'Chase Field', ATL: 'Truist Park', BAL: 'Camden Yards',
  BOS: 'Fenway Park', CHC: 'Wrigley Field', CIN: 'Great American Ball Park',
  CLE: 'Progressive Field', COL: 'Coors Field', CWS: 'Guaranteed Rate Field',
  DET: 'Comerica Park', HOU: 'Minute Maid Park', KC: 'Kauffman Stadium',
  LAA: 'Angel Stadium', LAD: 'Dodger Stadium', MIA: 'loanDepot Park',
  MIL: 'American Family Field', MIN: 'Target Field', NYM: 'Citi Field',
  NYY: 'Yankee Stadium', OAK: 'Oakland Coliseum', PHI: 'Citizens Bank Park',
  PIT: 'PNC Park', SD: 'Petco Park', SEA: 'T-Mobile Park',
  SF: 'Oracle Park', STL: 'Busch Stadium', TB: 'Tropicana Field',
  TEX: 'Globe Life Field', TOR: 'Rogers Centre', WSH: 'Nationals Park',
};

// Dome parks
const DOME_PARKS = ['ARI', 'HOU', 'MIA', 'MIL', 'SEA', 'TB', 'TEX', 'TOR'];

/**
 * Compute air density from temperature and humidity.
 * @param {number} tempF - Temperature in Fahrenheit
 * @param {number} rhum - Relative humidity (0-100)
 * @param {number} pres - Pressure in hPa (default 1013.25)
 * @returns {number} Air density in kg/m³
 */
function computeAirDensity(tempF, rhum, pres = 1013.25) {
  const tempK = (tempF - 32) * 5/9 + 273.15;
  const tempC = (tempF - 32) * 5/9;
  const eSat = 6.1078 * Math.pow(10, (7.5 * tempC) / (tempC + 237.3));
  const e = (rhum / 100) * eSat;
  const pD = (pres - e) * 100;
  return pD / (287.05 * tempK) + (e * 100) / (461.5 * tempK);
}

/**
 * Standardize a value using center and scale.
 * @param {number} value - Raw value
 * @param {Object} scaling - Object with center and scale properties
 * @returns {number} Standardized value
 */
function standardize(value, scaling) {
  if (!scaling) return value;
  return (value - scaling.center) / scaling.scale;
}

/**
 * Prediction model class for v6 weather-adjusted predictions.
 */
class PredictionModel {
  /**
   * Initialize the model with parameters from dashboard_params_v6.json.
   * @param {Object} params - Model parameters
   */
  constructor(params) {
    this.params = params;
    this.scaling = params.scaling || {};
    this.metadata = params._metadata || {};
    this.marineLayerParks = this.metadata.marine_layer_parks || MARINE_LAYER_PARKS;
    this.elevationByPark = this.metadata.elevation_by_park || {};
  }

  /**
   * Get elevation for a park.
   * @param {string} park - Park code
   * @returns {number} Elevation in feet
   */
  getElevation(park) {
    return this.elevationByPark[park] || 512.6; // League average
  }

  /**
   * Check if park is in marine layer zone.
   * @param {string} park - Park code
   * @returns {boolean}
   */
  isMarineLayerPark(park) {
    return this.marineLayerParks.includes(park);
  }

  /**
   * Check if park has a dome/retractable roof.
   * @param {string} park - Park code
   * @returns {boolean}
   */
  isDomePark(park) {
    return DOME_PARKS.includes(park);
  }

  /**
   * Make a prediction for the given park and weather conditions.
   * @param {string} park - Park code (e.g., 'SF', 'COL')
   * @param {Object} weather - Weather conditions
   * @param {number} weather.temp_f - Temperature in Fahrenheit
   * @param {number} weather.rhum - Relative humidity (0-100)
   * @param {number} weather.wspd_mph - Wind speed in mph
   * @param {number} weather.wind_cf - Wind component toward center field
   * @param {boolean} weather.is_night - Night game flag
   * @param {number} [weather.air_density] - Air density (computed if not provided)
   * @param {number} [weather.elevation_ft] - Elevation (looked up if not provided)
   * @param {string} target - 'strikeouts' or 'runs'
   * @returns {Object} Prediction result with components breakdown
   */
  predict(park, weather, target = 'strikeouts') {
    if (!this.params[target]) {
      throw new Error(`Unknown target: ${target}`);
    }
    if (!this.params[target].park_effects[park]) {
      throw new Error(`Unknown park: ${park}`);
    }

    const fe = this.params[target].fixed_effects;
    const parkEffects = this.params[target].park_effects;
    const parkWeatherInteractions = this.params[target].park_weather_interactions[park] || {};

    // Extract and prepare inputs
    const tempF = weather.temp_f || 72;
    const rhum = weather.rhum || 60;
    const wspdMph = weather.wspd_mph || 8;
    const windCf = weather.wind_cf || 0;
    const isNight = weather.is_night ? 1 : 0;
    const airDensity = weather.air_density || computeAirDensity(tempF, rhum);
    const elevationFt = weather.elevation_ft || this.getElevation(park);

    // Standardize values
    const std = {
      temp_f: standardize(tempF, this.scaling.temp_f),
      rhum: standardize(rhum, this.scaling.rhum),
      wspd_mph: standardize(wspdMph, this.scaling.wspd_mph),
      wind_cf: standardize(windCf, this.scaling.wind_cf),
      air_density: standardize(airDensity, this.scaling.air_density),
      elevation_ft: standardize(elevationFt, this.scaling.elevation_ft),
    };

    // Calculate individual contributions
    const contributions = {
      intercept: fe.Intercept || 0,
      temp_f: (fe.temp_f || 0) * std.temp_f,
      rhum: (fe.rhum || 0) * std.rhum,
      wspd_mph: (fe.wspd_mph || 0) * std.wspd_mph,
      wind_cf: (fe.wind_cf || 0) * std.wind_cf,
      air_density: (fe.air_density || 0) * std.air_density,
      elevation: (fe.elevation_ft || 0) * std.elevation_ft,
      is_night: (fe.is_night || 0) * isNight,
      park_effect: parkEffects[park] || 0,
    };

    // Marine layer effects
    const isMarine = this.isMarineLayerPark(park) ? 1 : 0;
    const isDay = 1 - isNight;
    contributions.marine_layer_day = (fe.marine_layer_day || 0) * isMarine * isDay;
    contributions.marine_layer_night = (fe.marine_layer_night || 0) * isMarine * isNight;

    // Park-weather interactions
    let interactionTotal = 0;
    for (const wf of ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']) {
      if (parkWeatherInteractions[wf] && std[wf] !== undefined) {
        interactionTotal += parkWeatherInteractions[wf] * std[wf];
      }
    }
    contributions.park_weather_interactions = interactionTotal;

    // Calculate total prediction
    const prediction = Object.values(contributions).reduce((sum, val) => sum + val, 0);

    // Group contributions
    const weatherEffects = {
      temperature: contributions.temp_f,
      humidity: contributions.rhum,
      wind_speed: contributions.wspd_mph,
      wind_direction: contributions.wind_cf,
      air_density: contributions.air_density,
      elevation: contributions.elevation,
    };

    const timeEffects = {
      day_night: contributions.is_night,
      marine_layer: contributions.marine_layer_day + contributions.marine_layer_night,
    };

    const parkEffectsGroup = {
      base_park_effect: contributions.park_effect,
      park_weather_interactions: contributions.park_weather_interactions,
    };

    return {
      prediction: Math.round(prediction * 100) / 100,
      target,
      park,
      parkName: PARK_NAMES[park] || park,
      isMarineLayerPark: this.isMarineLayerPark(park),
      isDomePark: this.isDomePark(park),
      rawInputs: {
        temp_f: tempF,
        rhum,
        wspd_mph: wspdMph,
        wind_cf: windCf,
        air_density: Math.round(airDensity * 10000) / 10000,
        elevation_ft: elevationFt,
        is_night: isNight,
      },
      components: {
        weather_effects: {
          ...Object.fromEntries(
            Object.entries(weatherEffects).map(([k, v]) => [k, Math.round(v * 10000) / 10000])
          ),
          subtotal: Math.round(Object.values(weatherEffects).reduce((a, b) => a + b, 0) * 10000) / 10000,
        },
        time_effects: {
          ...Object.fromEntries(
            Object.entries(timeEffects).map(([k, v]) => [k, Math.round(v * 10000) / 10000])
          ),
          subtotal: Math.round(Object.values(timeEffects).reduce((a, b) => a + b, 0) * 10000) / 10000,
        },
        park_effects: {
          ...Object.fromEntries(
            Object.entries(parkEffectsGroup).map(([k, v]) => [k, Math.round(v * 10000) / 10000])
          ),
          subtotal: Math.round(Object.values(parkEffectsGroup).reduce((a, b) => a + b, 0) * 10000) / 10000,
        },
        intercept: Math.round(contributions.intercept * 10000) / 10000,
      },
      contributions: Object.fromEntries(
        Object.entries(contributions).map(([k, v]) => [k, Math.round(v * 10000) / 10000])
      ),
    };
  }

  /**
   * Get elevation contribution analysis.
   * @param {string} park - Park code
   * @param {Object} weather - Weather conditions
   * @param {string} target - 'strikeouts' or 'runs'
   * @returns {Object} Elevation contribution details
   */
  getElevationContribution(park, weather, target = 'strikeouts') {
    const result = this.predict(park, weather, target);
    const elevationFt = this.getElevation(park);
    const leagueAvg = 512.6;

    return {
      park,
      parkName: PARK_NAMES[park] || park,
      elevation_ft: elevationFt,
      league_avg_elevation: leagueAvg,
      elevation_deviation: elevationFt - leagueAvg,
      contribution: result.contributions.elevation,
      interpretation: this._interpretElevation(elevationFt, result.contributions.elevation, target),
    };
  }

  /**
   * Get marine layer contribution analysis.
   * @param {string} park - Park code
   * @param {Object} weather - Weather conditions
   * @param {string} target - 'strikeouts' or 'runs'
   * @returns {Object} Marine layer contribution details
   */
  getMarineLayerContribution(park, weather, target = 'strikeouts') {
    const result = this.predict(park, weather, target);
    const isNight = weather.is_night || false;

    return {
      park,
      parkName: PARK_NAMES[park] || park,
      is_marine_layer_park: this.isMarineLayerPark(park),
      is_night_game: isNight,
      marine_layer_day_contribution: result.contributions.marine_layer_day,
      marine_layer_night_contribution: result.contributions.marine_layer_night,
      total_contribution: result.contributions.marine_layer_day + result.contributions.marine_layer_night,
      interpretation: this._interpretMarineLayer(park, isNight, result.contributions, target),
    };
  }

  /**
   * Generate elevation interpretation.
   * @private
   */
  _interpretElevation(elevationFt, contribution, target) {
    const targetUnit = target === 'strikeouts' ? 'strikeouts' : 'runs';
    let locationDesc;

    if (elevationFt > 4000) locationDesc = 'high altitude';
    else if (elevationFt > 1000) locationDesc = 'elevated';
    else if (elevationFt < 100) locationDesc = 'near sea level';
    else locationDesc = 'moderate elevation';

    let effectDesc;
    if (Math.abs(contribution) < 0.05) {
      effectDesc = `minimal effect on ${targetUnit}`;
    } else if (contribution > 0) {
      effectDesc = `adds ${contribution.toFixed(2)} ${targetUnit}`;
    } else {
      effectDesc = `reduces ${targetUnit} by ${Math.abs(contribution).toFixed(2)}`;
    }

    return `At ${elevationFt.toFixed(0)} ft (${locationDesc}), elevation ${effectDesc}.`;
  }

  /**
   * Generate marine layer interpretation.
   * @private
   */
  _interpretMarineLayer(park, isNight, contributions, target) {
    const targetUnit = target === 'strikeouts' ? 'strikeouts' : 'runs';

    if (!this.isMarineLayerPark(park)) {
      return `${PARK_NAMES[park] || park} is not in the marine layer zone - no fog effect.`;
    }

    const timeDesc = isNight ? 'night' : 'day';
    const contribution = isNight ? contributions.marine_layer_night : contributions.marine_layer_day;

    if (Math.abs(contribution) < 0.05) {
      return `Marine layer has minimal effect on ${targetUnit} for this ${timeDesc} game.`;
    } else if (contribution > 0) {
      return `Pacific coast fog conditions add ${contribution.toFixed(2)} ${targetUnit} for this ${timeDesc} game.`;
    } else {
      return `Pacific coast fog conditions reduce ${targetUnit} by ${Math.abs(contribution).toFixed(2)} for this ${timeDesc} game.`;
    }
  }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    PredictionModel,
    computeAirDensity,
    standardize,
    MARINE_LAYER_PARKS,
    DOME_PARKS,
    PARK_NAMES,
  };
}

// Export for ES6 modules
export {
  PredictionModel,
  computeAirDensity,
  standardize,
  MARINE_LAYER_PARKS,
  DOME_PARKS,
  PARK_NAMES,
};
