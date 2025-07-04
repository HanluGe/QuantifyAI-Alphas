from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open_ = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
#vwap = Feature(FeatureType.VWAP)
target = Ref(close, -24) / close - 1
# if interval == "60min":
#     target = Ref(close, -24) / close - 1
# elif interval == "5min":
#     target = Ref(close, -12) / close - 1