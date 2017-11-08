import os.path
from scipy.sparse import coo_matrix


def no_file():
    print("Dataset not found, please download the file and put it on : data/")

def fetch_amazonratingonly(min_rating=3):
	filePath = 'data/ratings_Books.csv'
	if not os.path.exists(filePath):
		return no_file()
	#Data to create for our coo_matrix
	data, row, col = [], [], []

	#Books by id and users
	books, users = {}, {}

    # Read the file and fill variables with data to
	# create the matrix and have the artists by id

	file = open(filePath)
	#File looks like this: (user,item,rating,timestamp)
	#AH2L9G3DQHHAJ,0000000116,4.0,1019865600
	#A2IIIDRK3PRRZY,0000000116,1.0,1395619200

	#The enumerate allows to loop over something and have an automatic counter
	for counter, line in enumerate(file):

		data_fromLine = line.split(',')
		user = data_fromLine[0]
		book = data_fromLine[1]
		rating = float(data_fromLine[2])
		#No particular uses of this information: timestamp = data_fromLine[3]
		if book not in books:
			books[book] = len(books)
		if user not in users:
			users[user] = len(users)

		if rating > min_rating:
			data.append(rating)
			row.append(users[user])
			col.append(books[book])

		#littlebreak to not hava a too long loop
		if counter > 10000:
			break
	#Our matrix  (rating, (users_id,books_id))
	coo = coo_matrix((data,(row,col)))

    # We return the matrix, the book dictionary and the amount of users
	dictionary = {
        'matrix' : coo,
        'books' : books,
        'users' : users
    }
	return dictionary