"""
time_series_model_prophet.py
Wrapper for using Prophet model with COVID-related timeseries
"""
import argparse
import logging
import os
import pickle
from collections import OrderedDict
from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse_utils import date_action
from fbprophet import Prophet
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from IPython import embed

from src import config
from src.stat_utils import ttest_confidence_interval_autocorr_correction

logging.basicConfig(format=config.LOGGING_FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ProphetWrapper:
    """
    Class wrapping the Prophet model, trains on pre timeseries, predicts post, and outputs results/graphs
    """
    UNCERTAINTY_SAMPLES = 10000

    def __init__(self, timeseries: pd.Series, start_date: date, prediction_date: date, end_date: date,
                 label: str, data_name: str, interval_width: float = 0.95, include_holidays: bool = True,
                 seasonality_mode: str = "additive", rolling_window_size: int = 7):
        """
        Initialize prophet model and save dates necessary for distinguishing time periods
        :param start_date: Beginning of pre-COVID period that is considered
        :param prediction_date: Beginning of post-COVID period/end of pre-COVID
        :param end_date: End date of defined post-COVID period
        :param data_name: A unique identifier for this CSV/column combo (for saving)
        :param interval_width: width of confidence interval
        :param include_holidays: whether or not to include US holidays in the prophet model
        """
        # use yearly seasonality, but not daily or weekly
        # daily = changes within day, but we are only considering daily data
        # weekly should be smoothed over using rolling mean
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                             interval_width=interval_width, uncertainty_samples=self.UNCERTAINTY_SAMPLES,
                             seasonality_mode=seasonality_mode)
        if include_holidays:
            # add US holidays if specified - the majority of Reddit users are in the US
            self.model.add_country_holidays(country_name="US")

        # store dates to be used
        self.pre_start = start_date
        self.pre_end = prediction_date - timedelta(days=1)
        self.post_start = prediction_date
        self.post_end = end_date

        self.rolling_window_size = rolling_window_size

        # validate and pre-process timeseries
        self.preprocessed_timeseries = self._preprocess_timeseries(timeseries)
        if self.preprocessed_timeseries.index[0] < self.pre_start:
            self.preprocessed_timeseries = self.preprocessed_timeseries[self.pre_start:]
        self.preprocessed_timeseries = self.preprocessed_timeseries[:self.post_end]
        self.pre_timeseries = self.preprocessed_timeseries[:self.pre_end]
        self.post_timeseries = self.preprocessed_timeseries[self.post_start:]

        # store label to be used when saving/creating figures
        self.label = label

        # store interval width to use in t-test
        self.interval_width = interval_width

        # name for output directory
        self.output_dir = os.path.join(
            "data", "prophet_output", data_name,
            f"{start_date}.{prediction_date}.{end_date}.{include_holidays}.{interval_width}.{seasonality_mode}")

        self.already_run = (os.path.exists(os.path.join(self.output_dir, "model.pickle"))
                            and os.path.exists(os.path.join(self.output_dir, "forecast.csv"))
                            and os.path.exists(os.path.join(self.output_dir, "preprocessed_timeseries.csv")))
        if self.already_run:
            logger.info("Script has already been run with these settings. Expecting that model, forecast, and "
                        "timeseries are saved. Will get missing metrics if needed.")

        # there is some randomness in the inference - use a seed, and also increase # of samples for more accuracy
        np.random.seed(1234)

    def fit_prophet_model(self) -> Prophet:
        """
        Fit a prophet model to our pre-timeseries (or load from disk)
        :return: None
        """
        if self.already_run:
            with open(os.path.join(self.output_dir, "model.pickle"), "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model.fit(self._convert_to_prophet_df(self.pre_timeseries))
        return self.model

    def forecast_and_save_results(self):
        """
        Forecast with a trained model, and save results to disk
        If script has already been run, just compute metrics/create plots
        Results to be saved:
        1. Dataframe of forecast
        2. Dataframe of the preprocessed timeseries itself, to use for comparison
        3. Dataframe of the model residuals
        4. Computed statistics for pre and post period, i.e. MAPE, % of points outside of CI, mean difference
           (to understand direction of change)
        5. Figure with predictions
        6. Figure with model components
        7. ACF plot for post-TS outside of CI
        8. PACF plot for post-TS outside of CI
        9. The model itself
        :return: None
        """
        if self.already_run:
            raw_forecast = pd.read_csv(os.path.join(self.output_dir, "forecast.csv"), parse_dates=["ds"])
            preprocessed_timeseries = pd.read_csv(os.path.join(self.output_dir, "preprocessed_timeseries.csv"),
                                                  index_col="Date", parse_dates=["Date"], squeeze=True)
            preprocessed_timeseries.index.name = "index"
            pd.testing.assert_series_equal(preprocessed_timeseries, self.preprocessed_timeseries)
        else:
            to_forecast_df = self._convert_to_prophet_df(self.preprocessed_timeseries)
            raw_forecast = self.model.predict(to_forecast_df)

        reindexed_forecast = raw_forecast.set_index("ds")
        pre_forecast = reindexed_forecast[:self.pre_end]
        post_forecast = reindexed_forecast[self.post_start:]

        # t-test on values outside of CI
        outside_ci_pre = np.array([int(not self._within_ci(reindexed_forecast, value, dt))
                                   for dt, value in self.pre_timeseries.items()])
        outside_ci_post = np.array([int(not self._within_ci(reindexed_forecast, value, dt))
                                    for dt, value in self.post_timeseries.items()])
        _, pval = ttest_confidence_interval_autocorr_correction(outside_ci_pre, outside_ci_post, self.interval_width)

        # get series of just anomalies dates - actual values and forecasted values
        post_timeseries_outlier = self.post_timeseries[outside_ci_post.astype(bool)]
        post_forecast_outlier = post_forecast['yhat'][outside_ci_post.astype(bool)]

        # compute some statistics about predictions:
        # 1. MAPE pre/post COVID
        # 2. % of points outside of confidence interval pre/post COVID
        # 3. Change in direction of the mean: are our predictions higher/lower than real values in post-COVID world?
        # 4. Change in direction of the mean for just anomalous observations
        result_computations_series = pd.Series(OrderedDict([
            ("MAPE pre-COVID", self._mean_absolute_percentage_error(self.pre_timeseries, pre_forecast["yhat"])),
            ("MAPE post-COVID", self._mean_absolute_percentage_error(self.post_timeseries, post_forecast["yhat"])),
            ("Outside CI pre-COVID", self._points_outside_ci(self.pre_timeseries, pre_forecast)),
            ("Outside CI post-COVID", self._points_outside_ci(self.post_timeseries, post_forecast)),
            ("Mean difference post-COVID", self._mean_difference(self.post_timeseries, post_forecast["yhat"])),
            ("Mean difference outliers post-COVID",
             self._mean_difference(post_timeseries_outlier, post_forecast_outlier)),
            ("PVal Outside CI post-COVID", pval),
        ]))

        # get the residuals (both pre and post)
        residuals = pd.concat((self.pre_timeseries, self.post_timeseries)) - reindexed_forecast["yhat"]

        # create graph of predictions
        fig_predictions = self._create_predictions_plot(raw_forecast)

        # create graph of model components
        fig_components = self._create_components_plot(raw_forecast)

        # prepare directory to save data
        os.makedirs(self.output_dir, exist_ok=True)

        # autocorrelation plots...
        if len(np.unique(outside_ci_post)) > 1:
            acf_plot = plot_acf(outside_ci_post, lags=10)
            pacf_plot = plot_pacf(outside_ci_post, lags=10)
            acf_plot.savefig(os.path.join(self.output_dir, "post_ci_acf.png"))
            pacf_plot.savefig(os.path.join(self.output_dir, "post_ci_pacf.png"))
            plt.close(acf_plot)
            plt.close(pacf_plot)

        # save data to directory
        if not self.already_run:
            # don't rewrite these if we have already run the script
            raw_forecast.to_csv(os.path.join(self.output_dir, "forecast.csv"), index=False)
            self.preprocessed_timeseries.to_csv(os.path.join(self.output_dir, "preprocessed_timeseries.csv"),
                                                index_label="Date")
            residuals.to_csv(os.path.join(self.output_dir, "residuals.csv"), index_label="Date")
            with open(os.path.join(self.output_dir, "model.pickle"), "wb") as f:
                pickle.dump(self.model, f)

        result_computations_series.to_csv(os.path.join(self.output_dir, "computed_results.csv"),
                                          index_label="Metric", header=["Result"])
        fig_predictions.savefig(os.path.join(self.output_dir, "predictions.png"))
        fig_components.savefig(os.path.join(self.output_dir, "model_components.png"))
        # close figures
        plt.close(fig_predictions)
        plt.close(fig_components)

    def _create_predictions_plot(self, forecast) -> Figure:
        """
        Plot forecast, adding in the real points in the forecasted time period, red = outside of CI, green = inside CI
        :param forecast: The forecast from Prophet
        :return: A matplotlib figure
        """
        fig = self.model.plot(forecast, xlabel="Date", ylabel=self.label)
        ax = fig.axes[0]
        reindexed_forecast = forecast.set_index("ds")

        # plot points: green if within confidence interval, red if outside
        point_colors = ["green" if self._within_ci(reindexed_forecast, value, dt) else "red"
                        for dt, value in self.post_timeseries.items()]
        self.post_timeseries.to_frame().reset_index().plot.scatter(
            x="index", y=self.label, color=point_colors, ax=ax, marker=".")
        return fig
    
    def _create_components_plot(self, forecast) -> Figure:
        """
        Plot components of prophet model, adding title
        :param forecast: The forecast from Prophet
        :return: A matplotlib figure
        """
        fig = self.model.plot_components(forecast)

        # relabel y axis to more readable "Date"
        fig.axes[0].set_xlabel("Date")

        # add overall title
        fig.axes[0].set_title(f"{self.label} Forecast Components")
        return fig

    def _preprocess_timeseries(self, timeseries: pd.Series) -> pd.Series:
        """
        Preprocess a timeseries. For now, this involves:
        1. Making sure that the time series is valid (has values for all necessary days)
        2. Computing rolling 7 day mean for smoothing purposes
        :param timeseries: timeseries indexed by dates
        :return: Preprocessed version of timeseries
        """
        if not self._validate_timeseries(timeseries):
            raise Exception("Data is not valid. Must have data for each day between start date and end date!")
        # question: do we want to include from the start date - 7 to have a rolling mean for the start date?
        # current implementation: we include the start date, but don't smooth with any data from before that date
        timeseries = timeseries[self.pre_start:].rolling(self.rolling_window_size, min_periods=1).mean().dropna()
        return timeseries

    def _validate_timeseries(self, timeseries: pd.Series):
        """
        confirm that the timeseries includes all dates from start_date through end_date
        overlap_range will have > 0 rows if we are missing dates from start_date to end_date
        :param timeseries: timeseries indexed by date
        :return: whether or not the timeseries is valid
        """
        overlap_range = pd.date_range(start=self.pre_start, end=self.post_end).difference(timeseries.index)
        return len(overlap_range) == 0

    @staticmethod
    def _convert_to_prophet_df(timeseries: pd.Series) -> pd.DataFrame:
        """
        Convert timeseries into format that is expected by Prophet
        :param timeseries: pandas series where index is a timestamp
        :return: dataframe where index is
        """
        return timeseries.to_frame().reset_index().rename(columns={"index": "ds", timeseries.name: "y"}).sort_values(
            by="ds")

    @staticmethod
    def _mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Computes mean absolute percent error
        :param y_true: series of true values
        :param y_pred: series of predicted values
        :return: mean absolute percent error
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def _points_outside_ci(self, y_true: pd.Series, forecast: pd.DataFrame) -> float:
        """
        Computes percentage of points outside of the confidence interval
        :param y_true: series of true values
        :param forecast: forecast from Prophet, containing results and CI
        :return: % of points that fall outside of confidence interval
        """
        inside_ci = sum(1 for dt in forecast.index if self._within_ci(forecast, y_true[dt], dt))
        return 100 * (len(forecast) - inside_ci) / len(forecast)

    @staticmethod
    def _mean_difference(y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Computes the mean difference between true and predicted values
        :param y_true: series of true values
        :param y_pred: series of predicted values
        :return: Mean difference between true and predicted values. Should represent the _direction_ of change
        """
        return (y_true - y_pred).mean()

    @staticmethod
    def _within_ci(forecast: pd.DataFrame, value: int, dt: datetime) -> bool:
        """
        Determine if a point is within the confidence interval from a forecast
        :param forecast: forecast from Prophet, containing results and CI
        :param value: the value to check
        :param dt: the date to check in the forecast
        :return: Whether or not the point falls in the CI
        """
        return forecast["yhat_lower"][dt] <= value <= forecast["yhat_upper"][dt]


# noinspection DuplicatedCode
def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files",
                        required=True,
                        nargs="+",
                        help="File(s) containing daily timeseries data (daily means, not preprocessed)")
    parser.add_argument("--data_column",
                        help="Name of column to analyze. If not specified, goes through all columns")
    parser.add_argument("--do_not_model_holidays",
                        action="store_true",
                        help="Do not include holidays in Prophet model")
    parser.add_argument("--start_date",
                        action=date_action(fmt="%Y-%m-%d"),
                        default=date(2017, 1, 1),
                        help="Start date for creating model of timeseries")
    parser.add_argument("--prediction_start",
                        action=date_action(fmt="%Y-%m-%d"),
                        default=date(2020, 3, 1),
                        help="Date of intervention "
                             "(default = beginning of when we see an effect of COVID on MH subreddits)")
    parser.add_argument("--end_date",
                        action=date_action(fmt="%Y-%m-%d"),
                        default=date(2020, 5, 31),
                        help="End date for date to be considered in predictions INCLUSIVE")
    parser.add_argument("--confidence_interval",
                        type=float,
                        default=0.95,
                        help="Confidence interval width to use with Prophet")
    parser.add_argument("--seasonality_mode",
                        choices=["additive", "multiplicative"],
                        default="additive",
                        help="Seasonality mode to use with Prophet")
    return parser.parse_args()


def main():
    """
    Read timeseries from CSV file, fit prophet model, save output
    """
    args = _parse_args()
    for data_file in args.data_files:
        dataframe = pd.read_csv(data_file, index_col=0, parse_dates=[0])

        if args.data_column is None:
            if dataframe.columns[-1] == "n_datapoints":
                data_columns = dataframe.columns[:-1]
            else:
                data_columns = dataframe.columns
        else:
            data_columns = [args.data_column]

        for data_column in data_columns:
            # define our timeseries: it is one column of the CSV that is passed in, determined by data_column
            timeseries = dataframe[data_column]

            # the prophet wrapper expects the index name to be "index"; rename to ensure that this is true
            timeseries.index.name = "index"

            # create data name - will be used when saving output
            data_name = f"{os.path.splitext(os.path.split(data_file)[1])[0]}.{data_column}"

            # create prophet model
            include_holidays = not args.do_not_model_holidays
            prophet_wrapper = ProphetWrapper(
                timeseries, args.start_date, args.prediction_start, args.end_date, data_column, data_name,
                interval_width=args.confidence_interval, include_holidays=include_holidays,
                seasonality_mode=args.seasonality_mode)
            prophet_wrapper.fit_prophet_model()
            prophet_wrapper.forecast_and_save_results()


if __name__ == "__main__":
    main()
