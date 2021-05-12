#! python3

'''
Hello Spark with pyspark
'''

# libs
import pyspark
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, udf
from pyspark.sql.types import FloatType, StringType, IntegerType



# Start Spark session
spark = SparkSession.builder.appName('rais_etl').getOrCreate()

# Load RAIS
rais = spark.read.load("data/RAIS_VINC_PUB_SUL.txt", format="csv", sep=";", inferSchema="true", header="true")
# type(rais)

# Change column names
newColumns = ["bairros_sp", "bairros_fortaleza", "bairros_rj", "causa_afastamento_1", "causa_afastamento_2", "causa_afastamento_3", "motivo_desligamento", "cbo_ocupacao_2002", "cnae_2_0_classe", "cnae_95_classe", "distritos_sp", "vinculo_ativo_31_12", "faixa_etaria", "faixa_hora_contrat", "faixa_remun_dezem_sm", "faixa_remun_media_sm", "faixa_tempo_emprego", "escolaridade_apos_2005", "qtd_hora_contr", "idade", "ind_cei_vinculado", "ind_simples", "mes_admissao", "mes_desligamento", "mun_trab", "municipio", "nacionalidade", "natureza_juridica", "ind_portador_defic", "qtd_dias_afastamento", "raca_cor", "regioes_adm_df", "vl_remun_dezembro_nom", "vl_remun_dezembro_sm", "vl_remun_media_nom", "vl_remun_media_sm", "cnae_2_0_subclasse", "sexo_trabalhador", "tamanho_estabelecimento", "tempo_emprego", "tipo_admissao", "tipo_estab", "tipo_estab_1", "tipo_defic", "tipo_vinculo", "ibge_subsetor", "vl_rem_janeiro_cc", "vl_rem_fevereiro_cc", "vl_rem_marco_cc", "vl_rem_abril_cc", "vl_rem_maio_cc", "vl_rem_junho_cc", "vl_rem_julho_cc", "vl_rem_agosto_cc", "vl_rem_setembro_cc", "vl_rem_outubro_cc", "vl_rem_novembro_cc", "ano_chegada_brasil", "ind_trab_intermitente", "ind_trab_parcial"] 
rais_sul = rais.toDF(*newColumns)

# Select fewer columns
rais_sul = rais_sul.select(["cnae_2_0_classe", "cnae_95_classe", "vinculo_ativo_31_12", "escolaridade_apos_2005", "qtd_hora_contr", "idade", "nacionalidade", "qtd_dias_afastamento", "raca_cor", "vl_remun_media_nom", "sexo_trabalhador", "tempo_emprego", "ibge_subsetor"])

# Function to convert commas to dots and turn the vector into Float type
commaToDot = udf(lambda x : float(str(x).replace(',', '.')), FloatType())

# Set data type
rais_sul = rais_sul.withColumn("cnae_2_0_classe", rais_sul["cnae_2_0_classe"].cast(StringType()))
rais_sul = rais_sul.withColumn("cnae_95_classe", rais_sul["cnae_95_classe"].cast(StringType()))
rais_sul = rais_sul.withColumn("vinculo_ativo_31_12", rais_sul["vinculo_ativo_31_12"].cast(StringType()))
rais_sul = rais_sul.withColumn("escolaridade_apos_2005", rais_sul["escolaridade_apos_2005"].cast(StringType()))
rais_sul = rais_sul.withColumn("qtd_hora_contr", rais_sul["qtd_hora_contr"].cast(IntegerType()))
rais_sul = rais_sul.withColumn("idade", rais_sul["idade"].cast(IntegerType()))
rais_sul = rais_sul.withColumn("nacionalidade", rais_sul["nacionalidade"].cast(StringType()))
rais_sul = rais_sul.withColumn("qtd_dias_afastamento", rais_sul["qtd_dias_afastamento"].cast(IntegerType()))
rais_sul = rais_sul.withColumn("raca_cor", rais_sul["raca_cor"].cast(StringType()))
rais_sul = rais_sul.withColumn('vl_remun_media_nom',commaToDot(rais_sul["vl_remun_media_nom"]))
rais_sul = rais_sul.withColumn("sexo_trabalhador", rais_sul["sexo_trabalhador"].cast(StringType()))
rais_sul = rais_sul.withColumn('tempo_emprego',commaToDot(rais_sul["tempo_emprego"]))
rais_sul = rais_sul.withColumn("ibge_subsetor", rais_sul["ibge_subsetor"].cast(StringType()))
# rais_sul.printSchema()
# rais_sul.show()

# Creating new columns
rais_sul = rais_sul.withColumn("vl_remu_ph", (rais_sul["vl_remun_media_nom"] / (rais_sul["qtd_hora_contr"]*4)).cast(FloatType()))

# Filters
rais_sul = rais_sul.filter(rais_sul.nacionalidade.isin(["10", "20"]) == True)
rais_sul = rais_sul.filter(rais_sul.vinculo_ativo_31_12 == "1")
rais_sul = rais_sul.filter(rais_sul.idade >= 18)
rais_sul = rais_sul.filter(rais_sul.qtd_hora_contr >= 12)
rais_sul = rais_sul.filter((rais_sul.vl_remun_media_nom > 1000) & (rais_sul.vl_remun_media_nom < 40000))

# Cleaned dataset
rais_sul.printSchema()
rais_sul.show()
print((rais_sul.count(), len(rais_sul.columns)))






# 
# CNAE = https://concla.ibge.gov.br/busca-online-cnae.html?view=subclasse&tipo=cnae&versao=10&subclasse=7220700