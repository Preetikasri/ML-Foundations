import graphlab
graphlab.product_key.set_product_key('1D68-D468-15C7-66B0-23E3-AC7E-C674-59CE')
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
graphlab.product_key.get_product_key()

sales = graphlab.SFrame('home_data.gl/')

myfeatures = ["bathrooms", "bedrooms","sqft_living",  "sqft_lot", "floors", "zipcode"]

sales[myfeatures].show()

sales.show(view ="BoxWhisker Plot", x= "zipcode", y="price")

sales[sales["zipcode"]=='98039']

q1 = sales[sales['zipcode']=='98039']
m1 = q1['price'].mean()

logic_1 = sales['sqft_living'] > 2000
logic_2 = sales['sqft_living'] < 4000
subset = sales[logic_1 & logic_2]
subset.num_rows()

print str(float(subset.num_rows())/sales.num_rows())

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', 
'grade', 
'waterfront', 
'view', 
'sqft_above', 
'sqft_basement', 
'yr_built', 
'yr_renovated', 
'lat', 'long', 
'sqft_living15', 
'sqft_lot15', 
]

train_data, test_data = sales.random_split(0.8, seed = 0)

mod1 = graphlab.linear_regression.create(train_data, target='price', features= my_features, validation_set=None)
mod2 = graphlab.linear_regression.create(train_data, target='price', features= advanced_features, validation_set=None)

mod2.evaluate(test_data)['rmse']-mod1.evaluate(test_data)['rmse']