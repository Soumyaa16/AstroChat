<script>
	let arr = ["Stargaze And Communicate :)"];
	let prompt = "";
  
	let getData = () => {
  
	  arr = [...arr, "You: " + prompt]
  
	  fetch("http://127.0.0.1:8000/echo/" + prompt)
		.then((r) => r.text())
		.then((t) => {
		  arr = [...arr, t.replaceAll('"', "")];
		});
	};
  </script>
  
  <div class="bg">
	<div class="msgs content">
	  {#each arr as i}
		<div class="msg">
		  {i}
		</div>
	  {/each}
	</div>
  
	<div class="content">
	  <div class="iarea">
		<input type="text" bind:value={prompt} on:submit={getData}/>
		<button on:click={getData}>Send</button>
	  </div>
	</div>
  </div>
  
  <style>
	input {
	  border-style: solid;
	  border-radius: 10px;
	  height: 5vh;
	  width: 90%;
	  background-color: rgb(91, 31, 92);
	  color: rgb(242, 234, 234);
	  font-size: 2.5vh;
	  padding: 0.1vh;
	}
  
	button{
	  background-color: #79548c;
	  border: none;
	  padding: 0.1vh;
	  height: 5vh;
	  width: 10%;
	  color: rgb(247, 243, 247);
	  border-radius: 10px;
	}
	.bg {
	  width: 80vw;
	  height: 100vh;
	  left: 0;
	  top: 0;
	  background-color: black;
  
	  display: flex;
	  flex-direction: column;
	  justify-content: space-between;
	}
  
	.content {
	  padding: 1vh 20vh 0.5vh 20vh;
	}
  
	.msgs {
	  display: flex;
	  flex-direction: column;
	  overflow-y: scroll;
  
	}
  
	.iarea {
	  padding-bottom: 7%;
	  display: flex;
	  flex-direction: row;
	  justify-content: space-between;
	  align-items: center;
	  gap: 10px;
	}
  
	.msg {
	  color: rgb(235, 219, 235);
	  padding: 1vh 2vw 1vh 2vw;
	  border-bottom: 2px solid #cccce7; 
	}
  </style>