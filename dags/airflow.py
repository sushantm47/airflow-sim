"""
DAG: Wholesale_Customer_Segmentation

load_data -> validate_data -> data_preprocessing -> build_save_model -> load_model (elbow)
    -> quality_gate --(pass)--> segment_customers -> pipeline_summary
                    --(fail)--> flag_low_quality  -> pipeline_summary
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src.lab import (
    build_save_model,
    data_preprocessing,
    load_data,
    load_model_elbow,
    quality_gate,
    segment_customers,
    validate_data,
)

default_args = {
    "owner": "sushant",
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="Wholesale_Customer_Segmentation",
    default_args=default_args,
    description="Segment wholesale customers by annual spending (K-Means + quality gate)",
    schedule=None,  # trigger manually from the UI
    catchup=False,
    tags=["mlops", "clustering", "customer-segmentation"],
    params={
        "max_k": Param(10, type="integer", minimum=3, maximum=15,
                       description="Largest number of segments to try"),
        "min_silhouette": Param(0.15, type="number", minimum=0, maximum=1,
                                description="Quality gate: minimum silhouette score to publish"),
    },
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=load_data,
    )

    validate_data_task = PythonOperator(
        task_id="validate_data_task",
        python_callable=validate_data,
        op_args=[load_data_task.output],
    )

    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing_task",
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id="build_save_model_task",
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
        op_kwargs={"max_k": "{{ params.max_k }}"},
    )

    load_model_task = PythonOperator(
        task_id="load_model_task",
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    quality_gate_task = BranchPythonOperator(
        task_id="quality_gate_task",
        python_callable=quality_gate,
        op_args=[build_save_model_task.output, load_model_task.output],
        op_kwargs={"min_silhouette": "{{ params.min_silhouette }}"},
    )

    segment_customers_task = PythonOperator(
        task_id="segment_customers_task",
        python_callable=segment_customers,
        op_args=["model.sav", load_model_task.output, build_save_model_task.output],
    )

    flag_low_quality_task = BashOperator(
        task_id="flag_low_quality_task",
        bash_command="echo 'WARNING: segments are not well separated - report NOT published. "
                     "Review the data or lower min_silhouette.'",
    )

    pipeline_summary_task = BashOperator(
        task_id="pipeline_summary_task",
        bash_command="echo 'Pipeline finished.' && ls -lh /opt/airflow/working_data",
        trigger_rule="none_failed_min_one_success",  # runs after whichever branch ran
    )

    load_data_task >> validate_data_task >> data_preprocessing_task
    data_preprocessing_task >> build_save_model_task >> load_model_task >> quality_gate_task
    quality_gate_task >> [segment_customers_task, flag_low_quality_task] >> pipeline_summary_task
