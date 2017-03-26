import csv

with open('text_emotion.csv') as file:
    with open('clean_text_emotions.csv', 'w') as csv_output:
        for line in file:
            # import pdb; pdb.set_trace()
            clean_separators = (line.replace(',\"','ó')
                                    .replace('\"ó','ó')
                                    .replace('\"',' ')
                                    .replace(',',' ')
                                    .replace('ó',',')
                                    .replace('\'',' '))
            no_extra_col = clean_separators[:-2] if clean_separators[-2]=="," else clean_separators

            single_spaced_rows = ' '.join(no_extra_col.split()) + "\n"

            csv_output.write(single_spaced_rows)
