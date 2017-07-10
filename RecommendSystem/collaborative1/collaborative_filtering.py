# Implementation of collaborative filtering recommendation engine

from recommendation_data import dataset
from math import sqrt

#For distance calculation

def similarity_score(person1, person2):
    # Returns ratio Euclidean distance score of person1 and person2

    both_viewed = {}  # To get both rated items by person1 and person2

    for item in dataset[person1]:
        if item in dataset[person2]:
            both_viewed[item] = 1

        # Conditions to check they both have an common rating items
        if len(both_viewed) == 0:
            return 0

        # Finding Euclidean distance
        sum_of_eclidean_distance = []

        for item in dataset[person1]:
            if item in dataset[person2]:
                sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item], 2))
        sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

        # Return the inverted value of euclidean distance
        return 1 / (1 + sqrt(sum_of_eclidean_distance))


def pearson_correlation(person1, person2):
    # To get both rated items
    both_rated = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    # Checking for number of ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each user
    person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

    # Sum up the squares of preferences of each user
    person1_square_preferences_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (
    person1_preferences_sum * person2_preferences_sum / number_of_ratings)
    denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (
    person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r


print similarity_score('Lisa Rose', 'Jack Matthews')

print pearson_correlation('Lisa Rose', 'Jack Matthews')

#Get ranking of people of most similar taste to user

def most_similar_users(person, number_of_users):
    # returns the number_of_users (similar persons) for a given specific person.
    scores = [(pearson_correlation(person, other_person), other_person) for other_person in dataset if
              other_person != person]

    # Sort the similar persons so that highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]

def most_similar_users2(person, number_of_users):
    # returns the number_of_users (similar persons) for a given specific person.
    scores = [(similarity_score(person, other_person), other_person) for other_person in dataset if
              other_person != person]

    # Sort the similar persons so that highest scores person will appear at the first
    scores.sort()
    return scores[0:number_of_users]

print 'by Pearson model, the following people are most similar to Lisa '
print most_similar_users('Lisa Rose', 3)

print 'by Eucludean model, the following people are most similar to Lisa'
print most_similar_users2('Lisa Rose', 3)

#recommend movies to a user by Pearson
def user_recommendations(person):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    for other in dataset:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:
                # Similarity * score
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendations_list = [recommend_item for score, recommend_item in rankings]
    return recommendations_list


print "Pearson Recommendations for Toby"
print user_recommendations('Toby')


#recommend movies to a user by Euclean
def user_recommendations2(person):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    for other in dataset:
        # don't compare me to myself
        if other == person:
            continue
        sim = similarity_score(person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:
                # Similarity * score
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendations_list = [recommend_item for score, recommend_item in rankings]
    return recommendations_list


print "Eucludean Recommendations for Toby"
print user_recommendations2('Toby')