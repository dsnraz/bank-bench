# -*- coding: utf-8 -*-
import sys 
sys.path.append('../memory_bank')
# from azure_client import LLMClientSimple
import openai, json, os
import argparse
import copy

class LLMClientSimple:

    def __init__(self, gen_config=None, handler=None):
        """handler: 可选的 BaseModelHandler 实例。提供后使用 handler 生成，否则走原有 OpenAI API 逻辑。"""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.disable_tqdm = False
        self.gen_config = gen_config
        self._handler = handler

    def generate_text_simple(self, prompt, prompt_num, language='en'):
        # 如果提供了 handler，使用 handler 生成（走本地模型或兼容 API）
        if self._handler is not None:
            return self._generate_with_handler(prompt, language)

        self.gen_config['n'] = prompt_num
        retry_times, count = 5, 0
        response = None
        while response is None and count < retry_times:
            try:
                request = copy.deepcopy(self.gen_config)
                if language == 'cn':
                    message = [
                        {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
                        {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
                        {"role": "system", "content": "好的，我会尽力帮你的。"},
                        {"role": "user", "content": f"{prompt}"}]
                else:
                    message = [
                        {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                        {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                        {"role": "system", "content": "Sure, I will do my best to assist you."},
                        {"role": "user", "content": f"{prompt}"}]
                response = openai.ChatCompletion.create(
                    **request, messages=message)
            except Exception as e:
                print(e)
                if 'This model\'s maximum context' in str(e):
                    cut_length = 1800 - 200 * (count)
                    print('max context length reached, cut to {}'.format(cut_length))
                    prompt = prompt[-cut_length:]
                    response = None
                count += 1
        if response:
            task_desc = response['choices'][0]['message']['content']
        else:
            task_desc = ''
        return task_desc

    def _generate_with_handler(self, prompt, language='en'):
        """通过外部 handler 生成，传递原始 messages 结构（角色与原版 OpenAI 调用完全一致）。"""
        cfg = self.gen_config or {}
        max_tokens = cfg.get('max_tokens', 400)
        temperature = cfg.get('temperature', 0.7)
        top_p = cfg.get('top_p', 1.0)
        frequency_penalty = cfg.get('frequency_penalty', 0.0)
        presence_penalty = cfg.get('presence_penalty', 0.0)
        if language == 'cn':
            messages = [
                {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
                {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
                {"role": "system", "content": "好的，我会尽力帮你的。"},
                {"role": "user", "content": f"{prompt}"}]
        else:
            messages = [
                {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                {"role": "system", "content": "Sure, I will do my best to assist you."},
                {"role": "user", "content": f"{prompt}"}]
        return self._handler.generate(prompt, messages=messages, max_new_tokens=max_tokens,
                                       temperature=temperature, top_p=top_p,
                                       frequency_penalty=frequency_penalty,
                                       presence_penalty=presence_penalty)
    

chatgpt_config = {"model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 400,
        "top_p": 1.0,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        "stop": ["<|im_end|>", "¬人类¬"]
        }

llm_client = LLMClientSimple(chatgpt_config)

def summarize_content_prompt(content,user_name,boot_name,language='en'):
    prompt = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：\n' if language=='cn' else 'Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n'
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_prompt(content,language='en'):
    prompt = '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n' if language=='cn' else "Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n"
    for date,summary_dict in content:
        summary = summary_dict['content']
        prompt += (f"\n时间{date}发生的事件为{summary.strip()}" if language=='cn' else f"At {date}, the events are {summary.strip()}")
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_personality(content,language='en'):
    prompt = '以下是用户在多段对话中展现出来的人格特质和心情，以及当下合适的回复策略：\n' if language=='cn' else "The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:"
    for date,summary in content:
        prompt += (f"\n在时间{date}的分析为{summary.strip()}" if language=='cn' else f"At {date}, the analysis shows {summary.strip()}")
    prompt += ('\n请总体概括用户的性格和AI恋人最合适的回复策略，尽量简洁精炼，高度概括。总结为：' if language=='cn' else "Please provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI lover, summarized as:")
    return prompt

def summarize_person_prompt(content,user_name,boot_name,language):
    prompt = f'请根据以下的对话推测总结{user_name}的性格特点和心情，并根据你的推测制定回复策略。对话内容：\n' if language=='cn' else f"Based on the following dialogue, please summarize {user_name}'s personality traits and emotions, and devise response strategies based on your speculation. Dialogue content:\n"
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        # prompt += f"\n用户：{query.strip()}"
        # prompt += f"\nAI：{response.strip()}"
        prompt += f"\n{user_name}：{query.strip()}"
        prompt += f"\n{boot_name}：{response.strip()}"

    prompt += (f'\n{user_name}的性格特点、心情、{boot_name}的回复策略为：' if language=='cn' else f"\n{user_name}'s personality traits, emotions, and {boot_name}'s response strategy are:")
    return prompt



def summarize_memory(memory_dir, name=None, language='cn',
                     extraction_handler=None, generation_handler=None):
    """
    对记忆库进行摘要提取。

    参数:
        extraction_handler: 可选 BaseModelHandler，用于逐日摘要和人格分析（Phase 1）。
        generation_handler: 可选 BaseModelHandler，用于整体历史和人格汇总（Phase 2）。
        两个 handler 均可选；不传时沿用原有 OpenAI API 逻辑。
    """
    boot_name = 'AI'
    gen_prompt_num = 1
    memory = json.loads(open(memory_dir, 'r', encoding='utf8').read())
    all_prompts, all_his_prompts, all_person_prompts = [], [], []

    # Phase 1: 使用 extraction_handler（逐日摘要）
    extraction_client = LLMClientSimple(chatgpt_config, handler=extraction_handler)
    # Phase 2: 使用 generation_handler（整体汇总），未提供则复用 extraction
    gen_handler = generation_handler or extraction_handler
    generation_client = LLMClientSimple(chatgpt_config, handler=gen_handler)

    for k, v in memory.items():
        if name is not None and k != name:
            continue
        user_name = k
        print(f'Updating memory for user {user_name}')
        if v.get('history') is None:
            continue
        history = v['history']
        if v.get('summary') is None:
            memory[user_name]['summary'] = {}
        if v.get('personality') is None:
            memory[user_name]['personality'] = {}
        for date, content in history.items():
            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True
            hisprompt = summarize_content_prompt(content, user_name, boot_name, language)
            person_prompt = summarize_person_prompt(content, user_name, boot_name, language)
            if his_flag:
                his_summary = extraction_client.generate_text_simple(
                    prompt=hisprompt, prompt_num=gen_prompt_num, language=language)
                memory[user_name]['summary'][date] = {'content': his_summary}
            if person_flag:
                person_summary = extraction_client.generate_text_simple(
                    prompt=person_prompt, prompt_num=gen_prompt_num, language=language)
                memory[user_name]['personality'][date] = person_summary

        overall_his_prompt = summarize_overall_prompt(
            list(memory[user_name]['summary'].items()), language=language)
        overall_person_prompt = summarize_overall_personality(
            list(memory[user_name]['personality'].items()), language=language)
        memory[user_name]['overall_history'] = generation_client.generate_text_simple(
            prompt=overall_his_prompt, prompt_num=gen_prompt_num, language=language)
        memory[user_name]['overall_personality'] = generation_client.generate_text_simple(
            prompt=overall_person_prompt, prompt_num=gen_prompt_num, language=language)

    with open(memory_dir, 'w', encoding='utf8') as f:
        print(f'Sucessfully update memory for {name}')
        json.dump(memory, f, ensure_ascii=False)
    return memory

if __name__ == '__main__':
    summarize_memory('../memories/eng_memory_cases.json',language='en')


                


