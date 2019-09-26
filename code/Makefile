# A simple Makefile that documents and automates the data pipeline and analyses
# with no real dependency management.

ALL = \
      overextension \
      vocab_and_lexicon \
      childes_noun_counts \
      gen_rel_freqs \
      visual_embeddings \
      filter_files \
      distance_matrix \
      analysis_production \
      analysis_cv_production \
      analysis_logistic_regression \
      analysis_mcdonough_production \
      analysis_comprehension \
      analysis_rate_of_overextension \
      analysis_category_evolution \
      analysis_mcdonough_frequencies \
      analysis_top_productions \
      plot_production \
      plot_logistic_regression \
      plot_mcdonough_production \
      plot_comprehension \
      analyses \
      plots \

.PHONY: all $(ALL)

all: $(ALL)

overextension:
	python3 scripts/generate_clean_overextension_pairs.py

vocab_and_lexicon:
	python3 scripts/generate_vocab_and_lexicon_files.py
	python3 scripts/generate_vocab_and_lexicon_files.py --mcdonough

childes_noun_counts:
	python3 scripts/gen_childes_noun_counts.py

gen_rel_freqs:
	python3 scripts/gen_vocab_freqs.py

visual_embeddings:
	python3 scripts/gen_visual_embeddings.py

filter_files:
	python3 scripts/filter_word_files_for_features.py
	python3 scripts/filter_word_files_for_features.py --mcdonough

distance_matrix:
	python3 scripts/precompute_distance_matrix.py
	python3 scripts/precompute_distance_matrix.py --mcdonough
	python3 scripts/precompute_distance_matrix.py --square
	python3 scripts/precompute_distance_matrix.py --square_mcdonough

analysis_production:
	python3 analyses/production.py --features 0,1,2
	python3 analyses/production.py --features 0,1
	python3 analyses/production.py --features 0,2
	python3 analyses/production.py --features 1,2
	python3 analyses/production.py --features 0
	python3 analyses/production.py --features 1
	python3 analyses/production.py --features 2
	python3 analyses/production.py --baseline
	python3 analyses/production.py --uniform --features 0,1,2
	python3 analyses/production.py --uniform --features 0,1
	python3 analyses/production.py --uniform --features 0,2
	python3 analyses/production.py --uniform --features 1,2
	python3 analyses/production.py --uniform --features 0
	python3 analyses/production.py --uniform --features 1
	python3 analyses/production.py --uniform --features 2
	python3 analyses/production.py --uniform --baseline
	cp results/production_training_result_frequency_0,1,2.pkl intermediate/model_parameters.pkl
	cp results/production_training_result_uniform_0,1,2.pkl intermediate/model_parameters_uniform.pkl
	python3 analyses/compute_production_bics.py

analysis_cv_production:
	python3 analyses/production.py --cv --features 0,1,2
	python3 analyses/production.py --cv --features 0,1
	python3 analyses/production.py --cv --features 0,2
	python3 analyses/production.py --cv --features 1,2
	python3 analyses/production.py --cv --features 0
	python3 analyses/production.py --cv --features 1
	python3 analyses/production.py --cv --features 2
	python3 analyses/production.py --cv --baseline

analysis_logistic_regression:
	python3 analyses/logistic_regression.py

analysis_mcdonough_production:
	python3 analyses/mcdonough_production.py
	python3 analyses/mcdonough_production.py --uniform

analysis_comprehension:
	python3 analyses/comprehension.py

analysis_rate_of_overextension:
	python3 analyses/rate_of_overextension.py

analysis_category_evolution:
	python3 analyses/category_evolution.py --concepts data/sample_animals.csv \
	    > results/category_animals.txt
	python3 analyses/category_evolution.py --concepts data/sample_fruits.csv \
	    > results/category_fruits.txt
	python3 analyses/category_evolution.py --concepts data/sample_vehicles.csv \
	    > results/category_vehicles.txt

analysis_mcdonough_frequencies:
	python3 analyses/mcdonough_frequencies.py

analysis_top_productions:
	python3 analyses/generate_top_productions.py
	python3 analyses/sample_top_productions.py

plot_production:
	python3 analyses/plot_production_curves.py
	python3 analyses/plot_individual_production_curves.py

plot_logistic_regression:
	python3 analyses/plot_logistic_regression.py

plot_mcdonough_production:
	python3 analyses/plot_mcdonough_production.py

plot_comprehension:
	python3 analyses/plot_comprehension.py

analyses: \
    analysis_production \
    analysis_cv_production \
    analysis_logistic_regression \
    analysis_mcdonough_production \
    analysis_comprehension \
    analysis_rate_of_overextension \
    analysis_category_evolution \
    analysis_mcdonough_frequencies \
    analysis_top_productions

plots: \
    plot_production \
    plot_logistic_regression \
    plot_mcdonough_production \
    plot_comprehension

