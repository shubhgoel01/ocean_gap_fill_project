import re

with open('ocean_gapfill_mc/src/ocean_gapfill_mc/pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    '    ensure_directories(config.output_directories())',
    '    if config.generate_outputs_and_logs:\n        ensure_directories(config.output_directories())'
)

text = text.replace(
    '    configure_logging(config.logs_dir)',
    '    configure_logging(config.logs_dir, enable_file_logging=config.generate_outputs_and_logs)'
)

pattern = r'(    logger\.info\(\"Step 12/13: saving logs and reports\"\).*?bloom_paths = generate_bloom_detection_from_script\(config\)\n)'

def replacer(match):
    block = match.group(1)
    indented = '\n'.join(['    ' + line if line.strip() else line for line in block.split('\n')])
    return '    if config.generate_outputs_and_logs:\n' + indented + '''    else:
        logger.info("Step 12/13: skipping outputs and reports saving as requested")
        dataset_paths = {}
        plot_paths = []
        annual_cycle_paths = {}
        bloom_paths = {}
'''

text = re.sub(pattern, replacer, text, flags=re.DOTALL)

with open('ocean_gapfill_mc/src/ocean_gapfill_mc/pipeline.py', 'w', encoding='utf-8') as f:
    f.write(text)
