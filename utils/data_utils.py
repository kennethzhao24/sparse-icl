import random

# apply few-shot template based on task
def apply_few_shot_template(task, few_shot_example, remaining_data, sampled_indices):
    
    if task in ['gsm8k', 'aime25_1', 'aime25_2']:
        for i in sampled_indices:
            data_point = remaining_data[i]
            few_shot_example += '\n Question:' + data_point['question'] +\
                    '\n Answer:' + data_point['answer'] + '\n\n'
            
    elif task in ['math500', 'omni_math', 'aime24', 'aime24_sky']: 
        for i in sampled_indices:
            data_point = remaining_data[i]
            few_shot_example += '\n Problem:' + data_point['problem'] +\
                '\n Solution:' + data_point['solution'] +\
                    '\n Answer:' + data_point['answer'] + '\n\n'

    elif task in ['amc23']: 
        for i in sampled_indices:
            data_point = remaining_data[i]
            few_shot_example += '\n Problem:' + data_point['problem'] +\
                    '\n Answer:' + data_point['answer'] + '\n\n'
            
    return few_shot_example


# Create few-shot examples
def create_few_shot_example(task, remaining_data, num_shots, index):
    few_shot_example = ""
    population = [number for number in range(len(remaining_data)) if number != index]
    # random sampling
    sampled_indices = random.sample(population, num_shots)
    few_shot_example = apply_few_shot_template(task, 
                                               few_shot_example, 
                                               remaining_data, 
                                               sampled_indices)
    few_shot_example += remaining_data[index]['problem']
    return few_shot_example.strip()
