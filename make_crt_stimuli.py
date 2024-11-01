from typing import Union
import pandas as pd

# Stimuli taken from OSF repo: https://osf.io/z6kmw

##### CRT1
crt1=[{'task': 'A pear and a fridge together cost $140. The pear costs $100 more than the fridge. How much does the fridge cost?',
  'total_cost': '140',
  'more': '100',
  'correct': '$20.0',
  'intuitive': '$40.0',
  'number': 1},
 {'task': 'A potato and a camera together cost $1.40. The potato costs $1 more than the camera. How much does the camera cost?',
  'total_cost': '1.40',
  'more': '1',
  'correct': '$0.20',
  'intuitive': '$0.40',
  'number': 2},
 {'task': 'A boat and a potato together cost $110. The boat costs $100 more than the potato. How much does the potato cost?',
  'total_cost': '110',
  'more': '100',
  'correct': '$5.0',
  'intuitive': '$10.0',
  'number': 3},
 {'task': 'A light bulb and a pan together cost $12. The light bulb costs $10 more than the pan. How much does the pan cost?',
  'total_cost': '12',
  'more': '10',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 4},
 {'task': 'A chair and a coat together cost $13. The chair costs $10 more than the coat. How much does the coat cost?',
  'total_cost': '13',
  'more': '10',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 5},
 {'task': 'A tube of toothpaste and a wallet together cost $54. The tube of toothpaste costs $50 more than the wallet. How much does the wallet cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 6},
 {'task': 'A coffee table and a mixing bowl together cost $52. The coffee table costs $50 more than the mixing bowl. How much does the mixing bowl cost?',
  'total_cost': '52',
  'more': '50',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 7},
 {'task': 'A microwave oven and a flower box together cost $51. The microwave oven costs $50 more than the flower box. How much does the flower box cost?',
  'total_cost': '51',
  'more': '50',
  'correct': '$0.50',
  'intuitive': '$1.0',
  'number': 8},
 {'task': 'A can of baby food and a helmet together cost $1.20. The can of baby food costs $1 more than the helmet. How much does the helmet cost?',
  'total_cost': '1.20',
  'more': '1',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 9},
 {'task': 'A sculpture and a box of lipstick together cost $13. The sculpture costs $10 more than the box of lipstick. How much does the box of lipstick cost?',
  'total_cost': '13',
  'more': '10',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 10},
 {'task': 'A pair of leggings and a body lotion together cost $54. The pair of leggings costs $50 more than the body lotion. How much does the body lotion cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 11},
 {'task': 'A trampoline and a box of batteries together cost $520. The trampoline costs $500 more than the box of batteries. How much does the box of batteries cost?',
  'total_cost': '520',
  'more': '500',
  'correct': '$10.0',
  'intuitive': '$20.0',
  'number': 12},
 {'task': 'A mouse and a can of baby food together cost $1.20. The mouse costs $1 more than the can of baby food. How much does the can of baby food cost?',
  'total_cost': '1.20',
  'more': '1',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 13},
 {'task': 'A bottle of shampoo and a knife together cost $110. The bottle of shampoo costs $100 more than the knife. How much does the knife cost?',
  'total_cost': '110',
  'more': '100',
  'correct': '$5.0',
  'intuitive': '$10.0',
  'number': 14},
 {'task': 'A bottle of bath salts and a set of crayons together cost $520. The bottle of bath salts costs $500 more than the set of crayons. How much does the set of crayons cost?',
  'total_cost': '520',
  'more': '500',
  'correct': '$10.0',
  'intuitive': '$20.0',
  'number': 15},
 {'task': 'A set of wine glasses and a pencil together cost $12. The set of wine glasses costs $10 more than the pencil. How much does the pencil cost?',
  'total_cost': '12',
  'more': '10',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 16},
 {'task': 'A wallet and a camera together cost $52. The wallet costs $50 more than the camera. How much does the camera cost?',
  'total_cost': '52',
  'more': '50',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 17},
 {'task': 'A pair of hiking boots and a coffee table together cost $1.20. The pair of hiking boots costs $1 more than the coffee table. How much does the coffee table cost?',
  'total_cost': '1.20',
  'more': '1',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 18},
 {'task': 'A tube of toothpaste and a pear together cost $12. The tube of toothpaste costs $10 more than the pear. How much does the pear cost?',
  'total_cost': '12',
  'more': '10',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 19},
 {'task': 'A trampoline and a flower box together cost $54. The trampoline costs $50 more than the flower box. How much does the flower box cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 20},
 {'task': 'A silicone case and a lawnmower together cost $130. The silicone case costs $100 more than the lawnmower. How much does the lawnmower cost?',
  'total_cost': '130',
  'more': '100',
  'correct': '$15.0',
  'intuitive': '$30.0',
  'number': 21},
 {'task': 'A toy and a microwave oven together cost $5.40. The toy costs $5 more than the microwave oven. How much does the microwave oven cost?',
  'total_cost': '5.40',
  'more': '5',
  'correct': '$0.20',
  'intuitive': '$0.40',
  'number': 22},
 {'task': 'A purse and a mixing bowl together cost $5.40. The purse costs $5 more than the mixing bowl. How much does the mixing bowl cost?',
  'total_cost': '5.40',
  'more': '5',
  'correct': '$0.20',
  'intuitive': '$0.40',
  'number': 23},
 {'task': 'A hat and a mouse together cost $540. The hat costs $500 more than the mouse. How much does the mouse cost?',
  'total_cost': '540',
  'more': '500',
  'correct': '$20.0',
  'intuitive': '$40.0',
  'number': 24},
 {'task': 'A purse and a box of cigarettes together cost $1.40. The purse costs $1 more than the box of cigarettes. How much does the box of cigarettes cost?',
  'total_cost': '1.40',
  'more': '1',
  'correct': '$0.20',
  'intuitive': '$0.40',
  'number': 25},
 {'task': 'A box of mascara and a microwave oven together cost $14. The box of mascara costs $10 more than the microwave oven. How much does the microwave oven cost?',
  'total_cost': '14',
  'more': '10',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 26},
 {'task': 'A bag and a food processor together cost $54. The bag costs $50 more than the food processor. How much does the food processor cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 27},
 {'task': 'A coat and a pot together cost $510. The coat costs $500 more than the pot. How much does the pot cost?',
  'total_cost': '510',
  'more': '500',
  'correct': '$5.0',
  'intuitive': '$10.0',
  'number': 28},
 {'task': 'A jacket and a set of wine glasses together cost $5.10. The jacket costs $5 more than the set of wine glasses. How much does the set of wine glasses cost?',
  'total_cost': '5.10',
  'more': '5',
  'correct': '$0.05',
  'intuitive': '$0.10',
  'number': 29},
 {'task': 'A hat and a coffee table together cost $1.20. The hat costs $1 more than the coffee table. How much does the coffee table cost?',
  'total_cost': '1.20',
  'more': '1',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 30},
 {'task': 'A pencil and a keyboard together cost $14. The pencil costs $10 more than the keyboard. How much does the keyboard cost?',
  'total_cost': '14',
  'more': '10',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 31},
 {'task': 'A bag and a coffee table together cost $53. The bag costs $50 more than the coffee table. How much does the coffee table cost?',
  'total_cost': '53',
  'more': '50',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 32},
 {'task': 'A mixing bowl and a laundry detergent together cost $52. The mixing bowl costs $50 more than the laundry detergent. How much does the laundry detergent cost?',
  'total_cost': '52',
  'more': '50',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 33},
 {'task': 'A perfume and a bottle of ouzo together cost $120. The perfume costs $100 more than the bottle of ouzo. How much does the bottle of ouzo cost?',
  'total_cost': '120',
  'more': '100',
  'correct': '$10.0',
  'intuitive': '$20.0',
  'number': 34},
 {'task': 'A bottle of bath salts and a pack of diapers together cost $510. The bottle of bath salts costs $500 more than the pack of diapers. How much does the pack of diapers cost?',
  'total_cost': '510',
  'more': '500',
  'correct': '$5.0',
  'intuitive': '$10.0',
  'number': 35},
 {'task': 'A CD and a pan together cost $13. The CD costs $10 more than the pan. How much does the pan cost?',
  'total_cost': '13',
  'more': '10',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 36},
 {'task': 'A rice cooker and a sculpture together cost $130. The rice cooker costs $100 more than the sculpture. How much does the sculpture cost?',
  'total_cost': '130',
  'more': '100',
  'correct': '$15.0',
  'intuitive': '$30.0',
  'number': 37},
 {'task': 'A pair of sunglasses and a toy together cost $13. The pair of sunglasses costs $10 more than the toy. How much does the toy cost?',
  'total_cost': '13',
  'more': '10',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 38},
 {'task': 'A speaker and a bottle of ouzo together cost $5.20. The speaker costs $5 more than the bottle of ouzo. How much does the bottle of ouzo cost?',
  'total_cost': '5.20',
  'more': '5',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 39},
 {'task': 'A set of crayons and a wallet together cost $5.20. The set of crayons costs $5 more than the wallet. How much does the wallet cost?',
  'total_cost': '5.20',
  'more': '5',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 40},
 {'task': 'A rug and a sculpture together cost $54. The rug costs $50 more than the sculpture. How much does the sculpture cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 41},
 {'task': 'A pair of sunglasses and a light bulb together cost $130. The pair of sunglasses costs $100 more than the light bulb. How much does the light bulb cost?',
  'total_cost': '130',
  'more': '100',
  'correct': '$15.0',
  'intuitive': '$30.0',
  'number': 42},
 {'task': 'A scarf and a ring together cost $1.20. The scarf costs $1 more than the ring. How much does the ring cost?',
  'total_cost': '1.20',
  'more': '1',
  'correct': '$0.10',
  'intuitive': '$0.20',
  'number': 43},
 {'task': 'A pair of leggings and a scarf together cost $52. The pair of leggings costs $50 more than the scarf. How much does the scarf cost?',
  'total_cost': '52',
  'more': '50',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 44},
 {'task': 'A wall clock and a light bulb together cost $1.30. The wall clock costs $1 more than the light bulb. How much does the light bulb cost?',
  'total_cost': '1.30',
  'more': '1',
  'correct': '$0.150',
  'intuitive': '$0.30',
  'number': 45},
 {'task': 'A blanket and a flashlight together cost $52. The blanket costs $50 more than the flashlight. How much does the flashlight cost?',
  'total_cost': '52',
  'more': '50',
  'correct': '$1.0',
  'intuitive': '$2.0',
  'number': 46},
 {'task': 'A keyboard and a pair of socks together cost $54. The keyboard costs $50 more than the pair of socks. How much does the pair of socks cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 47},
 {'task': 'A set of wine glasses and a pizza together cost $53. The set of wine glasses costs $50 more than the pizza. How much does the pizza cost?',
  'total_cost': '53',
  'more': '50',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 48},
 {'task': 'A pear and a bottle of ouzo together cost $54. The pear costs $50 more than the bottle of ouzo. How much does the bottle of ouzo cost?',
  'total_cost': '54',
  'more': '50',
  'correct': '$2.0',
  'intuitive': '$4.0',
  'number': 49},
 {'task': 'A pizza and a toy together cost $13. The pizza costs $10 more than the toy. How much does the toy cost?',
  'total_cost': '13',
  'more': '10',
  'correct': '$1.5',
  'intuitive': '$3.0',
  'number': 50}]

#### CRT2
crt2=[{'task': 'How long does it take 4 people to tailor 4 jackets, if it takes 7 people 7 hours to tailor 7 jackets?',
  'correct': '7 hours',
  'intuitive': '4 hours',
  'number': 1},
 {'task': 'How long does it take 4 washing machines to wash 4 loads of laundry, if it takes 8 washing machines 8 hours to wash 8 loads of laundry?',
  'correct': '8 hours',
  'intuitive': '4 hours',
  'number': 2},
 {'task': 'How long does it take 50 bees to pollinate 50 flowers, if it takes 60 bees 60 minutes to pollinate 60 flowers?',
  'correct': '60 minutes',
  'intuitive': '50 minutes',
  'number': 3},
 {'task': 'How long does it take 1 carpenter to make 1 chair, if it takes 5 carpenters 5 days to make 5 chairs?',
  'correct': '5 days',
  'intuitive': '1 day',
  'number': 4},
 {'task': 'How long does it take 10 ovens to bake 10 lasagnas, if it takes 60 ovens 60 minutes to bake 60 lasagnas?',
  'correct': '60 minutes',
  'intuitive': '10 minutes',
  'number': 5},
 {'task': 'How long does it take 1 researcher to publish 1 paper, if it takes 6 researchers 6 years to publish 6 papers?',
  'correct': '6 years',
  'intuitive': '1 year',
  'number': 6},
 {'task': 'How long does it take 30 cleaners to clean 30 rooms, if it takes 50 cleaners 50 hours to clean 50 rooms?',
  'correct': '50 hours',
  'intuitive': '30 hours',
  'number': 7},
 {'task': 'How long does it take 40 students to change 40 light bulbs, if it takes 70 students 70 minutes to change 70 light bulbs?',
  'correct': '70 minutes',
  'intuitive': '40 minutes',
  'number': 8},
 {'task': 'How long does it take 40 builders to build 40 houses, if it takes 50 builders 50 weeks to build 50 houses?',
  'correct': '50 weeks',
  'intuitive': '40 weeks',
  'number': 9},
 {'task': 'How long does it take 5 people to plant 5 trees, if it takes 6 people 6 minutes to plant 6 trees?',
  'correct': '6 minutes',
  'intuitive': '5 minutes',
  'number': 10},
 {'task': 'How long does it take 30 coffee machines to make 30 coffees, if it takes 40 coffee machines 40 minutes to make 40 coffees?',
  'correct': '40 minutes',
  'intuitive': '30 minutes',
  'number': 11},
 {'task': 'How long does it take 5 machines to pack 5 boxes of chocolates, if it takes 8 machines 8 minutes to pack 8 boxes of chocolates?',
  'correct': '8 minutes',
  'intuitive': '5 minutes',
  'number': 12},
 {'task': 'How long does it take 10 children to eat 10 boxes of chocolates, if it takes 50 children 50 minutes to eat 50 boxes of chocolates?',
  'correct': '50 minutes',
  'intuitive': '10 minutes',
  'number': 13},
 {'task': 'How long does it take 2 people to read 2 books, if it takes 4 people 4 weeks to read 4 books?',
  'correct': '4 weeks',
  'intuitive': '2 weeks',
  'number': 14},
 {'task': 'How long does it take 5 teams to renovate 5 houses, if it takes 8 teams 8 weeks to renovate 8 houses?',
  'correct': '8 weeks',
  'intuitive': '5 weeks',
  'number': 15},
 {'task': 'How long does it take 30 people to knit 30 pairs of socks, if it takes 40 people 40 weeks to knit 40 pairs of socks?',
  'correct': '40 weeks',
  'intuitive': '30 weeks',
  'number': 16},
 {'task': 'How long does it take 40 people to pick 40 fields of strawberries, if it takes 70 people 70 hours to pick 70 fields of strawberries?',
  'correct': '70 hours',
  'intuitive': '40 hours',
  'number': 17},
 {'task': 'How long does it take 6 programmers to write 6 lines of code, if it takes 7 programmers 7 hours to write 7 lines of code?',
  'correct': '7 hours',
  'intuitive': '6 hours',
  'number': 18},
 {'task': 'How long does it take 4 photographers to take 4 photos, if it takes 8 photographers 8 hours to take 8 photos?',
  'correct': '8 hours',
  'intuitive': '4 hours',
  'number': 19},
 {'task': 'How long does it take 1 painter to paint 1 painting, if it takes 8 painters 8 hours to paint 8 paintings?',
  'correct': '8 hours',
  'intuitive': '1 hour',
  'number': 20},
 {'task': 'How long does it take 50 writers to write 50 books, if it takes 70 writers 70 minutes to write 70 books?',
  'correct': '70 minutes',
  'intuitive': '50 minutes',
  'number': 21},
 {'task': 'How long does it take 2 cooks to cook 2 meals, if it takes 8 cooks 8 minutes to cook 8 meals?',
  'correct': '8 minutes',
  'intuitive': '2 minutes',
  'number': 22},
 {'task': 'How long does it take 50 doctors to examine 50 patients, if it takes 60 doctors 60 minutes to examine 60 patients?',
  'correct': '60 minutes',
  'intuitive': '50 minutes',
  'number': 23},
 {'task': 'How long does it take 2 drivers to change 2 tires, if it takes 7 drivers 7 minutes to change 7 tires?',
  'correct': '7 minutes',
  'intuitive': '2 minutes',
  'number': 24},
 {'task': 'How long does it take 2 farm workers to pick 2 apples, if it takes 8 farm workers 8 seconds to pick 8 apples?',
  'correct': '8 seconds',
  'intuitive': '2 seconds',
  'number': 25},
 {'task': 'How long does it take 4 freezers to freeze 4 liters of water, if it takes 6 freezers 6 hours to freeze 6 liters of water?',
  'correct': '6 hours',
  'intuitive': '4 hours',
  'number': 26},
 {'task': 'How long does it take 20 bakers to bake 20 cakes, if it takes 80 bakers 80 hours to bake 80 cakes?',
  'correct': '80 hours',
  'intuitive': '20 hours',
  'number': 27},
 {'task': 'How long does it take 30 hair stylists to finish 30 hairstyles, if it takes 50 hair stylists 50 minutes to finish 50 hairstyles?',
  'correct': '50 minutes',
  'intuitive': '30 minutes',
  'number': 28},
 {'task': 'How long does it take 2 mechanics to fix 2 cars, if it takes 3 mechanics 3 hours to fix 3 cars?',
  'correct': '3 hours',
  'intuitive': '2 hours',
  'number': 29},
 {'task': 'How long does it take 20 tailors to make 20 dresses, if it takes 50 tailors 50 hours to make 50 dresses?',
  'correct': '50 hours',
  'intuitive': '20 hours',
  'number': 30},
 {'task': 'How long does it take 3 painters to paint 3 rooms, if it takes 4 painters 4 hours to paint 4 rooms?',
  'correct': '4 hours',
  'intuitive': '3 hours',
  'number': 31},
 {'task': 'How long does it take 40 trees to grow 40 leaves, if it takes 60 trees 60 days to grow 60 leaves?',
  'correct': '60 days',
  'intuitive': '40 days',
  'number': 32},
 {'task': 'How long does it take 1 runner to clean 1 shoe, if it takes 5 runners 5 minutes to clean 5 shoes?',
  'correct': '5 minutes',
  'intuitive': '1 minute',
  'number': 33},
 {'task': 'How long does it take 3 translators to translate 3 pages, if it takes 6 translators 6 hours to translate 6 pages?',
  'correct': '6 hours',
  'intuitive': '3 hours',
  'number': 34},
 {'task': 'How long does it take 2 machines to make 2 smartphones, if it takes 3 machines 3 hours to make 3 smartphones?',
  'correct': '3 hours',
  'intuitive': '2 hours',
  'number': 35},
 {'task': 'How long does it take 50 people to smoke 50 cigarettes, if it takes 70 people 70 minutes to smoke 70 cigarettes?',
  'correct': '70 minutes',
  'intuitive': '50 minutes',
  'number': 36},
 {'task': 'How long does it take 7 opticians to make 7 glasses, if it takes 8 opticians 8 days to make 8 glasses?',
  'correct': '8 days',
  'intuitive': '7 days',
  'number': 37},
 {'task': 'How long does it take 40 pipes to fill 40 containers, if it takes 60 pipes 60 hours to fill 60 containers?',
  'correct': '60 hours',
  'intuitive': '40 hours',
  'number': 38},
 {'task': 'How long does it take 50 people to eat 50 pizzas, if it takes 80 people 80 minutes to eat 80 pizzas?',
  'correct': '80 minutes',
  'intuitive': '50 minutes',
  'number': 39},
 {'task': 'How long does it take 20 kettles to boil 20 liters of water, if it takes 80 kettles 80 hours to boil 80 liters of water?',
  'correct': '80 hours',
  'intuitive': '20 hours',
  'number': 40},
 {'task': 'How long does it take 40 air conditioners to cool 40 rooms, if it takes 80 air conditioners 80 minutes to cool 80 rooms?',
  'correct': '80 minutes',
  'intuitive': '40 minutes',
  'number': 41},
 {'task': 'How long does it take 5 students to finish 5 exams, if it takes 7 students 7 minutes to finish 7 exams?',
  'correct': '7 minutes',
  'intuitive': '5 minutes',
  'number': 42},
 {'task': 'How long does it take 40 men to make 40 pies, if it takes 70 men 70 minutes to make 70 pies?',
  'correct': '70 minutes',
  'intuitive': '40 minutes',
  'number': 43},
 {'task': 'How long does it take 1 barista to make 1 coffee, if it takes 8 baristas 8 minutes to make 8 coffees?',
  'correct': '8 minutes',
  'intuitive': '1 minute',
  'number': 44},
 {'task': 'How long does it take 10 people to cook 10 packs of spaghetti, if it takes 70 people 70 minutes to cook 70 packs of spaghetti?',
  'correct': '70 minutes',
  'intuitive': '10 minutes',
  'number': 45},
 {'task': 'How long does it take 5 people to renovate 5 bathrooms, if it takes 8 people 8 days to renovate 8 bathrooms?',
  'correct': '8 days',
  'intuitive': '5 days',
  'number': 46},
 {'task': 'How long does it take 1 fish to eat 1 worm, if it takes 3 fish 3 days to eat 3 worms?',
  'correct': '3 days',
  'intuitive': '1 day',
  'number': 47},
 {'task': 'How long does it take 1 operator to connect 1 phone call, if it takes 3 operators 3 hours to connect 3 phone calls?',
  'correct': '3 hours',
  'intuitive': '1 hour',
  'number': 48},
 {'task': 'How long does it take 20 printers to print 20 documents, if it takes 80 printers 80 minutes to print 80 documents?',
  'correct': '80 minutes',
  'intuitive': '20 minutes',
  'number': 49},
 {'task': 'How long does it take 70 husbands to feed 70 babies, if it takes 80 husbands 80 minutes to feed 80 babies?',
  'correct': '80 minutes',
  'intuitive': '70 minutes',
  'number': 50}]

######### CRT3
crt3=[{'task': "In a city, a virus is spreading, causing the total number of infected individuals to double each day. If it takes 6 days for the entire city's population to be infected, how many days would it require for half of the people to become infected?",
  't': '6 days',
  'correct': '5 days',
  'intuitive': '3 days',
  'number': 1},
 {'task': "A pandemic is occurring in a state where the total number of infected individuals doubles daily. If it takes 10 days for the entire state's population to become infected, how many days would it take for half of the state's population to be infected?",
  't': '10 days',
  'correct': '9 days',
  'intuitive': '5 days',
  'number': 2},
 {'task': 'People are escaping from war. Each day, the total count of refugees doubles. If it takes 22 days for the entire population to evacuate, how long would it take for half of the population to do so?',
  't': '22 days',
  'correct': '21 days',
  'intuitive': '11 days',
  'number': 3},
 {'task': 'A farmer is plowing a field. Each hour, the total plowed area doubles. If it takes 10 hours for the entire field to be plowed, how long would it take for half of the field to be plowed?',
  't': '10 hours',
  'correct': '9 hours',
  'intuitive': '5 hours',
  'number': 4},
 {'task': 'The apples are dropping from an apple tree, with the total count of fallen apples doubling each day. If it requires 16 days for all the apples to drop, how many days would it take for half the apples to fall?',
  't': '16 days',
  'correct': '15 days',
  'intuitive': '8 days',
  'number': 5},
 {'task': 'Fish are migrating and each day, the total distance they cover doubles. If it takes the fish 18 days to reach their destination, how many days would it take for them to cover half the distance?',
  't': '18 days',
  'correct': '17 days',
  'intuitive': '9 days',
  'number': 6},
 {'task': 'A tree branch is falling, and with each passing second, the total distance it covered doubles. If it takes 6 seconds for the branch to reach the ground, how long would it take for it to cover one-half of the total distance?',
  't': '6 seconds',
  'correct': '5 seconds',
  'intuitive': '3 seconds',
  'number': 7},
 {'task': 'A fly is traveling from point A to point B. With each passing hour, the total distance it covered doubles. If the fly reaches point B in 12 hours, how long does it take for the fly to cover half of the distance?',
  't': '12 hours',
  'correct': '11 hours',
  'intuitive': '6 hours',
  'number': 8},
 {'task': 'A new concrete pavement is drying. The overall area of dried concrete doubles each day. If it requires 4 days for the entire pavement to dry completely, how many days does it take for half the pavement to become dry?',
  't': '4 days',
  'correct': '3 days',
  'intuitive': '2 days',
  'number': 9},
 {'task': 'There is a freezer filled with food, and the total volume of the frozen food doubles every hour. If it takes 16 hours for all the food to become frozen, how long would it take to freeze half of the food?',
  't': '16 hours',
  'correct': '15 hours',
  'intuitive': '8 hours',
  'number': 10},
 {'task': 'A pot of water is boiling on the stove, and with each passing hour, the overall volume of the evaporated water doubles. If the entire pot takes 6 hours to evaporate completely, how long does it take for half of the pot to evaporate?',
  't': '6 hours',
  'correct': '5 hours',
  'intuitive': '3 hours',
  'number': 11},
 {'task': 'There is a section of mold on a bread loaf that doubles in size every hour. If it takes 16 hours for the mold to completely cover the bread, how much time is needed for the mold to cover half of the bread?',
  't': '16 hours',
  'correct': '15 hours',
  'intuitive': '8 hours',
  'number': 12},
 {'task': 'A forest is engulfed in flames. Each day, the overall area of the scorched forest doubles in size. If it takes 18 days for the entire forest to be consumed by the fire, how many days would it take for half of the forest to be burnt?',
  't': '18 days',
  'correct': '17 days',
  'intuitive': '9 days',
  'number': 13},
 {'task': 'In a cave, there is a colony of bats with a daily population doubling. Given that it takes 60 days for the entire cave to be filled with bats, how many days would it take for the cave to be half-filled with bats?',
  't': '60 days',
  'correct': '59 days',
  'intuitive': '30 days',
  'number': 14},
 {'task': 'A section of grass is expanding within a garden, with the total area it occupies doubling daily. If it requires 12 days for the entire garden to be encompassed by the grass, how many days are needed for the grass to cover half of the garden?',
  't': '12 days',
  'correct': '11 days',
  'intuitive': '6 days',
  'number': 15},
 {'task': 'An investor possesses 1 bitcoin. Each day, their number of bitcoins doubles. If it takes them 30 days to achieve their investment target, how long would it take for them to reach half of that target?',
  't': '30 days',
  'correct': '29 days',
  'intuitive': '15 days',
  'number': 16},
 {'task': 'Fish inhabit a creek, and their population doubles each week. If it requires 24 weeks for the entire creek to become completely filled with fish, how long would it take to fill half of the creek with fish?',
  't': '24 weeks',
  'correct': '23 weeks',
  'intuitive': '12 weeks',
  'number': 17},
 {'task': 'A dust cloud hovers above the city, doubling in size each day. If it takes 12 days for the entire city to be engulfed by the cloud, how many days does it take for the cloud to cover half of the city?',
  't': '12 days',
  'correct': '11 days',
  'intuitive': '6 days',
  'number': 18},
 {'task': 'There is a sick student in the class. Each day, the number of sick students doubles. If it takes 6 days for the entire class to become sick, how many days does it take for half of the class to become sick?',
  't': '6 days',
  'correct': '5 days',
  'intuitive': '3 days',
  'number': 19},
 {'task': 'A colony of bacteria is growing on yogurt, and the size of the colony doubles daily. If it takes 4 days for the colony to completely cover the yogurt, how many days does it take for the patch to cover half of the yogurt?',
  't': '4 days',
  'correct': '3 days',
  'intuitive': '2 days',
  'number': 20},
 {'task': 'A moss patch is expanding on a rock, doubling its size each day. It takes 300 days for the moss to completely cover the rock. How many days are required for the moss to cover half of the rock?',
  't': '300 days',
  'correct': '299 days',
  'intuitive': '150 days',
  'number': 21},
 {'task': 'Beneath a tree, there is a heap of leaves that doubles in size every week. If it takes 4 weeks for the pile to attain a height of 4 meters, how much time is required for it to reach a height of 2 meters?',
  't': '4 weeks',
  'correct': '3 weeks',
  'intuitive': '2 weeks',
  'number': 22},
 {'task': 'Mushrooms are cultivated in a container, and their quantity doubles daily. Given that it takes 6 days for the mushrooms to fill the entire container, determine the time needed to fill half of the container.',
  't': '6 days',
  'correct': '5 days',
  'intuitive': '3 days',
  'number': 23},
 {'task': 'There is a man who raises rabbits in a barn. Each year, the rabbit population doubles. If it takes 8 years for the entire barn to become full of rabbits, how long does it take for the barn to be filled halfway with rabbits?',
  't': '8 years',
  'correct': '7 years',
  'intuitive': '4 years',
  'number': 24},
 {'task': 'Within a forest, there is a growing patch of ramson that doubles in size each week. If it takes 10 weeks for the entire forest to be covered with ramson, how long would it take for the ramson to cover half of the forest?',
  't': '10 weeks',
  'correct': '9 weeks',
  'intuitive': '5 weeks',
  'number': 25},
 {'task': 'It is currently raining, causing the lake to fill up with water. The volume of water in the lake doubles every day. If it takes a total of 20 days for the lake to become completely filled, how many days would it take for the lake to reach halfway to being full?',
  't': '20 days',
  'correct': '19 days',
  'intuitive': '10 days',
  'number': 26},
 {'task': 'Within a forest, there exists a tree that doubles its height each year. If the tree attains its maximum height in 10 years, determine the time it would take to achieve half of that maximum height.',
  't': '10 years',
  'correct': '9 years',
  'intuitive': '5 years',
  'number': 27},
 {'task': 'There is a flood occurring in a field. With each passing hour, the size of the flood-stricken area doubles. If it takes 20 hours for the entire field to become submerged, how many hours would it take for half of the field to be inundated?',
  't': '20 hours',
  'correct': '19 hours',
  'intuitive': '10 hours',
  'number': 28},
 {'task': 'A barrel is being filled with whiskey, and the total volume of whiskey doubles every minute. If it takes 12 minutes to completely fill the barrel, how long would it take to fill half of it with whiskey?',
  't': '12 minutes',
  'correct': '11 minutes',
  'intuitive': '6 minutes',
  'number': 29},
 {'task': 'Programmers are in the process of developing new software and each month the total quantity of written code doubles. If it requires 10 months to complete the entire code, how much time is needed to write half of the code?',
  't': '10 months',
  'correct': '9 months',
  'intuitive': '5 months',
  'number': 30},
 {'task': 'A factory is busy filling bags with chocolate cookies. The total number of bags filled doubles with each passing hour. Given that it takes 8 hours to fill all the bags, how much time is required to fill half of the bags?',
  't': '8 hours',
  'correct': '7 hours',
  'intuitive': '4 hours',
  'number': 31},
 {'task': 'An orange tree is sprouting leaves. The number of leaves doubles every month. Given that it takes 6 months for the entire tree to be covered with leaves, how many months does it take for the tree to be half covered with leaves?',
  't': '6 months',
  'correct': '5 months',
  'intuitive': '3 months',
  'number': 32},
 {'task': 'An iceberg is forming and its surface doubles every year. If the iceberg takes 10 years to grow to 10 square miles, how many years are required for it to grow to 5 square miles?',
  't': '10 years',
  'correct': '9 years',
  'intuitive': '5 years',
  'number': 33},
 {'task': 'A woman is in the process of growing her hair. Each year, the length of her hair doubles. If it takes 6 years for her hair to achieve a length of two meters, how long would it take for it to reach a length of one meter?',
  't': '6 years',
  'correct': '5 years',
  'intuitive': '3 years',
  'number': 34},
 {'task': 'Ants are crawling on a cake, and with each passing minute, their population on the cake doubles. If the entire cake is covered with ants in 30 minutes, how long would it take for half of the cake to be covered with ants?',
  't': '30 minutes',
  'correct': '29 minutes',
  'intuitive': '15 minutes',
  'number': 35},
 {'task': 'In a room, there are rats whose population doubles each month. If it takes 9 months for the rats to completely fill the room, how long would it take for them to occupy half of the room?',
  't': '9 months',
  'correct': '8 months',
  'intuitive': '4.5 months',
  'number': 36},
 {'task': 'The wood is burning in a fireplace, and the temperature doubles every minute. If it takes 20 minutes for the fireplace to reach a temperature of 600 degrees, how long does it take for the temperature to reach 300 degrees?',
  't': '20 minutes',
  'correct': '19 minutes',
  'intuitive': '10 minutes',
  'number': 37},
 {'task': 'In a fish tank, some algae are present. Each day, the quantity of algae multiplies by two. If it requires 30 days for the entire fish tank to become filled with algae, how much time is needed for half of the fish tank to be filled with algae?',
  't': '30 days',
  'correct': '29 days',
  'intuitive': '15 days',
  'number': 38},
 {'task': 'An elderly woman regularly feeds cats. Each day, the number of cats she feeds doubles. If she feeds 64 cats on the 6th day, on which day does she feed 32 cats?',
  't': '6th day',
  'correct': '5th day',
  'intuitive': '3rd day',
  'number': 39},
 {'task': 'Bamboo is growing in the garden, and its height doubles each day. It takes 5 days to reach a height of 10 meters. How long does it take for the bamboo to attain a height of 5 meters?',
  't': '5 days',
  'correct': '4 days',
  'intuitive': '2.5 days',
  'number': 40},
 {'task': 'A patient has been diagnosed with cancer, and the number of cancer cells doubles each week. If it takes 4 weeks for the cancer cells to reach a critical amount, how long will it take for the cells to reach half of that critical amount?',
  't': '4 weeks',
  'correct': '3 weeks',
  'intuitive': '2 weeks',
  'number': 41},
 {'task': 'There is a heap of fruit that is decaying. The quantity of spoiled fruit doubles daily. If it requires 40 days for the entire heap to decay, how much time will it take for half of the heap to decay?',
  't': '40 days',
  'correct': '39 days',
  'intuitive': '20 days',
  'number': 42},
 {'task': "During winter, the lake is gradually freezing. Each day, the ice covering the lake's surface doubles in size. If it takes 10 days for the entire surface of the lake to freeze, how long does it take for half of the lake's surface to become frozen?",
  't': '10 days',
  'correct': '9 days',
  'intuitive': '5 days',
  'number': 43},
 {'task': 'During winter, snowfall is occurring and the depth of the snow cover doubles every hour. If it takes 10 hours for the snow depth to reach 2 meters, how much time is required for it to reach a depth of 1 meter?',
  't': '10 hours',
  'correct': '9 hours',
  'intuitive': '5 hours',
  'number': 44},
 {'task': 'Five individuals are constructing a home. Each week, the cumulative amount of bricks they place doubles. Given that it takes them 8 weeks to lay all the bricks, determine the duration required for them to lay half of the bricks.',
  't': '8 weeks',
  'correct': '7 weeks',
  'intuitive': '4 weeks',
  'number': 45},
 {'task': "Two painters are working on painting a house. With every hour that passes, the total area they've painted doubles. If it takes 16 hours to paint the entire house, how many hours would it take for them to paint half of the house?",
  't': '16 hours',
  'correct': '15 hours',
  'intuitive': '8 hours',
  'number': 46},
 {'task': "A grandmother is knitting a scarf for her grandson. Each week, the scarf's length increases twofold. If it takes 6 weeks to complete the entire scarf, how long would it take to knit half of the scarf?",
  't': '6 weeks',
  'correct': '5 weeks',
  'intuitive': '3 weeks',
  'number': 47},
 {'task': 'A forest is expanding on an island, with the area occupied by the trees doubling each year. If it takes 140 years for the entire island to be covered with trees, how many years would it take for the forest to cover half of the island?',
  't': '140 years',
  'correct': '139 years',
  'intuitive': '70 years',
  'number': 48},
 {'task': 'A gas cylinder is experiencing a leak. With each passing hour, the total quantity of leaked gas doubles. If it requires 4 hours for the entire gas to leak out, how much time is needed for half of the gas to escape?',
  't': '4 hours',
  'correct': '3 hours',
  'intuitive': '2 hours',
  'number': 49},
 {'task': 'People are walking into a theater and taking their seats. The number of people in the room multiplies by two every minute. If it takes 6 minutes for all of the seats to be taken, how long does it take for half of the seats to be taken?',
  't': '6 minutes',
  'correct': '5 minutes',
  'intuitive': '3 minutes',
  'number': 50}]

def format_as_money(num: Union[str, int, float]) -> str:
    """
    Format a number value (as string, int, or float) into $X.XX format.
    This is useful in the tasks involving string comparison of dollar amounts.
    """
    if isinstance(num, str):
        if num.startswith("$"):
            num = float(num[1:])
        else:
            num = float(num)
        return '${:,.2f}'.format(num)
    else:
        return None

def save_crt_stimuli() -> None:
    # Combine all stimuli into one dataframe.
    stim_sets = {
        "crt1": crt1,
        "crt2": crt2,
        "crt3": crt3
    }
    for stim_set_name, stim_set in stim_sets.items():
        print(stim_set_name)
        df = pd.DataFrame(stim_set)
        df = df.rename(columns={
            "task": "question",
            "number": "item_id"
        })
        df["task"] = stim_set_name

        # Reorder some columns.
        meta_cols = ["task", "item_id", "question"]
        answer_cols = [c for c in list(df) if c not in meta_cols]
        col_order =  meta_cols + answer_cols
        df = df[col_order]

        # Format CRT1 answer options in standardized dollar format.
        if stim_set_name == "crt1":
            for i, row in df.iterrows():
                for col in answer_cols:
                    val = row[col]
                    dollar_str = format_as_money(val)
                    df.loc[i, f"{col}_money_fmt"] = dollar_str
    
        print(df.head())
        df.to_csv(f"data/stimuli/{stim_set_name}.csv", index=False)


if __name__ == "__main__":
    save_crt_stimuli()