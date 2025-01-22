async def run(self):
    self.gen_silent_mel()
    self.gen_silent_ids()

    while self.parent.is_valid():
        try:
            # Handle interaction speech interruptions
            if (not self._is_running_nlp and
                self.parent.runtime_data.speech_linked_list.head is not None and
                not self.parent.runtime_data.queue_interaction.empty()):
                
                interaction_speech = self.parent.runtime_data.queue_interaction.get()

                # Process interaction speech (either a list or a single speech)
                new_node = self.process_interaction_speech(interaction_speech)

                # If no new node was created, skip to the next iteration
                if new_node is None:
                    continue

                logger.info(f'2  交互话术打断处理:{interaction_speech.speech_id}')

                # Insert the new node into the appropriate location
                find_node = self.find_insert_node(interaction_speech)
                if find_node:
                    self.insert_node(find_node, new_node, interaction_speech)
                else:
                    logger.error(f'找不到未生成视频的节点：{interaction_speech.speech_id}')

            # Handle preset speech retrieval
            elif self.parent.runtime_data.get_unpushed_duration() < config.g_cache_unpushed_duration:
                logger.info('4 预设话术取出流程')
                logger.debug(f'get_unpushed_duration: {self.parent.runtime_data.get_unpushed_duration()}')
                self.gen_silent()

            await asyncio.sleep(tick_time)

        except Exception:
            logger.error(traceback.format_exc())

def process_interaction_speech(self, interaction_speech):
    """Processes interaction speech input, either as a single item or a list."""
    if isinstance(interaction_speech, list):
        return self.create_interaction_node_list(interaction_speech)
    else:
        logger.debug(f"silent speech audio_url is : {interaction_speech.audio_url}, video url is {interaction_speech.video_url}")
        return self.handle_single_interaction(interaction_speech)

def create_interaction_node_list(self, speech_list):
    """Creates a linked list from a list of speech data."""
    new_node = None
    end_node = None
    for speech_data in speech_list:
        node = Node(speech_data)
        if new_node is None:
            new_node = node
            end_node = new_node
        else:
            end_node.insert_node(node)
            end_node = end_node.tail
    self.log_interaction_list(new_node)
    return new_node

def log_interaction_list(self, new_node):
    """Logs the interaction speech list for debugging."""
    debug_str = '收到交互话术列表'
    debug_node = new_node
    while debug_node is not None:
        debug_str += f'{debug_node.data.speech_id} '
        debug_node = debug_node.tail
    logger.debug(debug_str)

def handle_single_interaction(self, interaction_speech):
    """Handles a single interaction speech input."""
    if interaction_speech.audio_url == '':
        self.mark_last_node_as_end()
        return None
    return Node(interaction_speech)

def mark_last_node_as_end(self):
    """Marks the last node as an end node."""
    last_node = self.parent.runtime_data.current_node.last()
    if last_node:
        last_node.data.is_end = True

def find_insert_node(self, interaction_speech):
    """Finds the node where the new speech should be inserted."""
    last_node = self.parent.runtime_data.current_node
    priority_node = self.get_highest_priority_node(last_node)
    
    if interaction_speech.priority == 1:
        return priority_node or last_node

    # For priority 2, find the next node
    if last_node.tail:
        last_node = last_node.tail
    
    # Search for a node that hasn't generated video yet
    return self.get_non_generated_video_node(last_node)

def get_highest_priority_node(self, last_node):
    """Gets the highest priority node from the linked list."""
    priority_node = None
    while last_node is not None:
        if last_node.data.priority > 0:
            priority_node = last_node
        last_node = last_node.tail
    return priority_node

def get_non_generated_video_node(self, last_node):
    """Searches for the next node that hasn't generated video."""
    while last_node is not None:
        if not last_node.data.is_wav2lip:
            logger.debug(f"寻找未生成视频的节点, break: {last_node.data.speech_id}")
            return last_node
        last_node = last_node.tail
    return None

def insert_node(self, find_node, new_node, interaction_speech):
    """Inserts the new node into the linked list at the appropriate position."""
    find_node_id = find_node.data.speech_id
    find_node_duration = config.g_cache_interation_duration
    check_node = self.parent.runtime_data.current_node
    need_cache_duration = self.calculate_needed_cache_duration(check_node, find_node_id, find_node_duration)

    # Insert the new node
    if new_node.data.priority == 2:
        find_node.tail = None  # Discard subsequent nodes

    if "silent" in find_node.data.speech_id:
        self.insert_silent_node(find_node, new_node)
    else:
        find_node.insert_node(new_node)

    self.parent.runtime_data.debug_speech_status('cache: insert interaction')

def calculate_needed_cache_duration(self, check_node, find_node_id, find_node_duration):
    """Calculates the duration needed to cache before inserting a new node."""
    need_cache_duration = 0
    while check_node is not None:
        need_cache_duration += (check_node.data.duration - check_node.data.elapse)
        if check_node.data.speech_id == find_node_id and need_cache_duration >= find_node_duration:
            return need_cache_duration
        check_node = check_node.tail
    return need_cache_duration

def insert_silent_node(self, find_node, new_node):
    """Inserts a silent node before the new node."""
    import copy
    silent_node = copy.deepcopy(find_node)
    silent_node.data.next_node_speak = True
    silent_node.data.speech_id = f"{silent_node.data.speech_id}_5"
    find_node.tail = silent_node
    silent_node.insert_node(new_node)
